"""
pipeline/queue.py
─────────────────
Simple asyncio.Semaphore-based job queue to prevent server overload.
Limits concurrent Remotion renders to one at a time.
"""

import asyncio

from db.supabase import update_job

_render_semaphore = asyncio.Semaphore(1)
_queue: list[str] = []
_queue_lock = asyncio.Lock()


async def _refresh_queue_positions() -> None:
    for index, queued_job_id in enumerate(_queue, start=1):
        await update_job(queued_job_id, status=f"queued (position {index})")


async def enqueue_job(job_id: str, frames: list[dict], app_description: str):
    """
    Enqueue a launch video job and execute it with a global concurrency limit of 1.
    """
    async with _queue_lock:
        _queue.append(job_id)
        await _refresh_queue_positions()

    print(f"[queue] Job {job_id} queued at position {_queue.index(job_id) + 1}")

    try:
        async with _render_semaphore:
            async with _queue_lock:
                if job_id in _queue:
                    _queue.remove(job_id)
                    await _refresh_queue_positions()

            from main import run_pipeline

            print(f"[queue] Job {job_id} starting pipeline")
            return await run_pipeline(job_id, frames, app_description)
    finally:
        async with _queue_lock:
            if job_id in _queue:
                _queue.remove(job_id)
                await _refresh_queue_positions()
