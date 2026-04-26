import os
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client


def _get_client() -> Client | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def update_job_status(job_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
    """
    Upsert job status in Supabase table `video_jobs`.
    If env vars are absent, this becomes a no-op for easy local dev.
    """
    client = _get_client()
    if client is None:
        return

    payload: dict[str, Any] = {
        "id": job_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    client.table("video_jobs").upsert(payload).execute()
