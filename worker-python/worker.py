# worker.py
import os
import time
import asyncio
import signal
from dotenv import load_dotenv
from bullmq import Worker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from rich.console import Console
import torch

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
REDIS_URL = os.getenv('REDIS_URL')

console = Console()

# DB connection
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})
Session = sessionmaker(bind=engine)

with Session() as session:
    session.execute(text("SELECT 1"))
console.print("Supabase connected")

for i in range(torch.cuda.device_count()):
    console.print(f"GPU {i}:", torch.cuda.get_device_name(i))

# -------------------------------------------------
async def process_job(job):
    job_id = job.data["jobId"]
    source_url = job.data["sourceUrl"]
    console.print(f"[bold blue]Worker {os.getpid()} processing job {job_id}[/] → {source_url}")

    # Mark as PROCESSING
    with Session() as session:
        session.execute(
            text('UPDATE "Jobs" SET status = \'PROCESSING\', progress = 10 WHERE id = :id'),
            {"id": job_id}
        )
        session.commit()

    # Simulate heavy GPU work (replace with YOLO, etc.)
    for i in range(1, 11):
        time.sleep(2)
        progress = 10 + i * 9
        with Session() as session:
            session.execute(
                text('UPDATE "Jobs" SET progress = :p WHERE id = :id'),
                {"p": progress, "id": job_id}
            )
            session.commit()

    # Mark as DONE
    fake_csv = f"https://example.com/results/{job_id}.csv"
    with Session() as session:
        session.execute(
            text('UPDATE "Jobs" SET status = \'DONE\', progress = 100, "outputCsvUrl" = :csv WHERE id = :id'),
            {"id": job_id, "csv": fake_csv}
        )
        session.commit()

    console.print(f"[bold green]Job {job_id} DONE → {fake_csv}[/]")

# -------------------------------------------------
async def main():
    # Create an event that will be triggered for shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        console.print("[yellow]Signal received, shutting down.[/]")
        shutdown_event.set()

    # Assign signal handlers to SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = Worker(
        "videoJobs",
        process_job,
        {"connection": REDIS_URL}
    )

    console.print("[bold magenta]Worker started – waiting for jobs (Ctrl+C to stop)[/]")

    # Wait until the shutdown event is set
    await shutdown_event.wait()

    # Close the worker
    console.print("[yellow]Cleaning up worker...[/]")
    await worker.close()
    console.print("[green]Worker shut down successfully.[/]")

if __name__ == "__main__":
    asyncio.run(main())