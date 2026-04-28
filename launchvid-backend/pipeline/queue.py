"""
pipeline/queue.py
─────────────────
Simple asyncio.Semaphore-based job queue to prevent server overload.
Limits concurrent Remotion renders to one at a time.
"""

import asyncio
from typing import Callable, Any

# FIX 6 applied: Concurrency limit using Semaphore
_render_semaphore = asyncio.Semaphore(1)
_job_queue: list[dict] = []
_job_counter = 0


async def enqueue_job(
    job_id: str,
    job_coro: Callable[[], Any],
    on_position_update: Callable[[int], Any] | None = None,
) -> Any:
    """
    Enqueue a job to run sequentially with other jobs.
    
    Args:
        job_id: Unique job identifier
        job_coro: Async coroutine to execute (e.g., run_pipeline(...))
        on_position_update: Optional callback to notify of queue position
    
    Returns:
        Result of the job coroutine
    """
    global _job_counter
    
    _job_counter += 1
    position = _job_counter

    # Add to queue
    _job_queue.append({"job_id": job_id, "position": position})

    try:
        # Notify of queue position
        if on_position_update:
            queue_position = len([j for j in _job_queue if j["position"] <= position]) - 1
            await on_position_update(queue_position)

        # Wait for semaphore (our turn)
        print(f"[queue] Job {job_id} queued at position {position}, waiting for render slot...")
        async with _render_semaphore:
            # Remove from queue now that we're about to run
            _job_queue[:] = [j for j in _job_queue if j["job_id"] != job_id]

            print(f"[queue] Job {job_id} starting render (position {position})")
            # Execute the actual job
            result = await job_coro
            print(f"[queue] Job {job_id} completed")
            return result

    except Exception as e:
        print(f"[queue] Job {job_id} failed: {e}")
        raise
    finally:
        # Ensure cleanup
        _job_queue[:] = [j for j in _job_queue if j["job_id"] != job_id]


def get_queue_position(job_id: str) -> int:
    """Get the current queue position of a job (0 = running, 1+ = waiting)."""
    for i, job in enumerate(_job_queue):
        if job["job_id"] == job_id:
            return i
    return -1  # Not in queue
