"""Video read-ahead and MP4/NVENC writers shared by full and peeing-only pipelines."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from typing import Protocol

import cv2
import numpy as np
from rich.console import Console

from settings import FFMPEG_PATH, NVENC_CQ, NVENC_PRESET, OUTPUT_VIDEO_ENCODER

console = Console()


class VideoWriterSink(Protocol):
    """Common surface for OpenCV ``VideoWriter`` and ffmpeg-backed writers."""

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class NullVideoWriterSink:
    """Discard frames (threshold sweeps without encoding)."""

    def write(self, frame: np.ndarray) -> None:
        del frame

    def release(self) -> None:
        pass


class _AsyncVideoWriterSink:
    """Bounded queue + background thread so NVENC/ffmpeg ``write`` does not stall inference."""

    def __init__(self, inner: VideoWriterSink, max_queue: int) -> None:
        self._inner = inner
        qsize = max(1, int(max_queue))
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=qsize)
        self._thr = threading.Thread(
            target=self._loop, name="pipeline-async-video-writer", daemon=True
        )
        self._thr.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            self._inner.write(item)

    def write(self, frame: np.ndarray) -> None:
        self._q.put(frame)

    def release(self) -> None:
        self._q.put(None)
        self._thr.join(timeout=600.0)
        self._inner.release()


class _ReadAheadVideoCapture:
    """Decode thread with a bounded queue ahead of the processing loop."""

    def __init__(self, cap: cv2.VideoCapture, max_queue: int) -> None:
        self._cap = cap
        qsize = max(1, int(max_queue))
        self._q: queue.Queue[tuple[bool, np.ndarray] | None] = queue.Queue(maxsize=qsize)

        def worker() -> None:
            try:
                while True:
                    ret, frame = self._cap.read()
                    self._q.put((ret, frame))
                    if not ret:
                        break
            finally:
                try:
                    self._q.put(None, timeout=5.0)
                except Exception:
                    try:
                        self._q.put_nowait(None)
                    except Exception:
                        pass

        self._stream_ended = False
        self._thr = threading.Thread(target=worker, name="pipeline-read-ahead", daemon=True)
        self._thr.start()

    def read(self) -> tuple[bool, np.ndarray]:
        if self._stream_ended:
            return False, np.array([], dtype=np.uint8)
        item = self._q.get()
        if item is None:
            self._stream_ended = True
            return False, np.array([], dtype=np.uint8)
        return item

    def get(self, prop: int) -> float:
        return float(self._cap.get(prop))

    def release(self) -> None:
        try:
            if self._thr.is_alive():
                self._thr.join(timeout=30.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())


def _maybe_wrap_video_sink(sink: VideoWriterSink, *, queue_size: int) -> VideoWriterSink:
    if queue_size <= 0:
        return sink
    return _AsyncVideoWriterSink(sink, queue_size)


def _maybe_wrap_capture(
    cap: cv2.VideoCapture, *, queue_size: int
) -> cv2.VideoCapture | _ReadAheadVideoCapture:
    if queue_size <= 0:
        return cap
    return _ReadAheadVideoCapture(cap, queue_size)


def _ffmpeg_nvenc_smoke_test(ffmpeg_bin: str) -> bool:
    """Return True if ``ffmpeg`` can run one frame through ``h264_nvenc`` (driver + build)."""
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.04",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


class _Cv2Mp4vSink:
    """OpenCV ``mp4v`` into MP4 (CPU MPEG-4 Part 2). Used only when ``OUTPUT_VIDEO_ENCODER=mp4v``."""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self._w = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self._w.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {path!r}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def release(self) -> None:
        self._w.release()


class _FfmpegNvencSink:
    """Stream raw BGR frames into ``ffmpeg`` ``h264_nvenc`` (GPU encoder, usually much faster)."""

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_bin: str,
        preset: str,
        cq: int,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            preset,
            "-rc",
            "vbr",
            "-cq",
            str(int(cq)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdin is None:
            self._proc.kill()
            raise RuntimeError("ffmpeg did not provide a stdin pipe")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match encoder {self._height}x{self._width}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
            self._proc.stdin = None
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc} while finishing {self._proc.args!r}")


def _open_output_video_sink(
    output_path: str,
    *,
    fps: float,
    width: int,
    height: int,
    encoder_mode: str | None = None,
) -> tuple[VideoWriterSink, str]:
    """Open a video sink: ``auto``/``nvenc`` require working ffmpeg ``h264_nvenc``; ``mp4v`` is explicit CPU only."""
    mode = (encoder_mode or OUTPUT_VIDEO_ENCODER or "auto").strip().lower()
    p = FFMPEG_PATH.strip()
    ffmpeg_bin = shutil.which(p)
    if ffmpeg_bin is None and os.path.isabs(p) and os.path.isfile(p) and os.access(p, os.X_OK):
        ffmpeg_bin = p

    if mode == "mp4v":
        sink_mp4v: VideoWriterSink = _Cv2Mp4vSink(output_path, fps, width, height)
        return sink_mp4v, "OpenCV VideoWriter mp4v (CPU)"

    if mode not in ("auto", "nvenc"):
        console.print(
            f"[red]OUTPUT_VIDEO_ENCODER must be 'auto', 'nvenc', or 'mp4v'; got {mode!r}[/]"
        )
        raise SystemExit(3)

    if not ffmpeg_bin:
        console.print(
            "[red]OUTPUT_VIDEO_ENCODER requires ffmpeg on PATH (see FFMPEG_PATH in settings.py).[/]"
        )
        raise SystemExit(3)

    if not _ffmpeg_nvenc_smoke_test(ffmpeg_bin):
        console.print(
            "[red]h264_nvenc smoke test failed (ffmpeg NVENC not usable on this host).[/]\n"
            f"  ffmpeg={ffmpeg_bin!r}\n"
            "  Install a GPU-enabled ffmpeg build and a working NVIDIA driver, or set OUTPUT_VIDEO_ENCODER='mp4v' explicitly for CPU encoding."
        )
        raise SystemExit(3)

    try:
        sink: VideoWriterSink = _FfmpegNvencSink(
            output_path,
            fps,
            width,
            height,
            ffmpeg_bin=ffmpeg_bin,
            preset=NVENC_PRESET,
            cq=NVENC_CQ,
        )
        return (
            sink,
            f"ffmpeg h264_nvenc (preset={NVENC_PRESET!r}, cq={NVENC_CQ})",
        )
    except Exception as exc:
        console.print(
            "[red]ffmpeg h264_nvenc writer failed (no fallback).[/]\n"
            f"  [dim]{type(exc).__name__}: {exc}[/]"
        )
        raise SystemExit(3) from exc
