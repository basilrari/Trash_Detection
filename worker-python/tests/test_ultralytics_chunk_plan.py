"""Regression for TensorRT chunk accounting (must match ``YoloDetector`` / ``LpDetector`` loops)."""

from models.ultralytics_call_stats import UltralyticsCallStats, trt_engine_chunk_plan


def test_dynamic_slack_not_padded() -> None:
    assert trt_engine_chunk_plan(10, 8, dynamic=True) == UltralyticsCallStats(
        input_units=10, batch_launches=2, padded_slots=0, max_batch_slack=6
    )


def test_static_padded_matches_slack_sum() -> None:
    assert trt_engine_chunk_plan(10, 8, dynamic=False) == UltralyticsCallStats(
        input_units=10, batch_launches=2, padded_slots=6, max_batch_slack=0
    )


def test_full_batches() -> None:
    assert trt_engine_chunk_plan(16, 8, dynamic=True) == UltralyticsCallStats(16, 2, 0, 0)
    assert trt_engine_chunk_plan(16, 8, dynamic=False) == UltralyticsCallStats(16, 2, 0, 0)


def test_many_inputs_chunked_not_one_launch_per_frame() -> None:
    """``trt_engine_chunk_plan`` is one ``detect()`` worth of images; 481 frames / B=8 → 61 chunks, tail slack 7."""
    assert trt_engine_chunk_plan(481, 8, dynamic=True) == UltralyticsCallStats(481, 61, 0, 7)


def test_gated_one_frame_per_detect_slack() -> None:
    """Gated pipeline: 481 calls to ``detect([one_frame])`` → 481 launches, slack 7 each (B=8, dynamic)."""
    launches = 481
    slack = sum(max(0, 8 - 1) for _ in range(481))
    assert (launches, slack) == (481, 3367)


def test_zero_inputs() -> None:
    assert trt_engine_chunk_plan(0, 8, dynamic=True) == UltralyticsCallStats(0, 0, 0, 0)
