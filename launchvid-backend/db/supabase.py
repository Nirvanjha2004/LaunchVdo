from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from datetime import datetime
from typing import Optional

# ── Client ─────────────────────────────────────────────────────────────────────

_client: Optional[Client] = None

def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env\n"
                "Get them from: https://supabase.com → your project → Settings → API"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── SQL to run once in Supabase SQL editor ─────────────────────────────────────
#
# create table if not exists jobs (
#   id           text primary key,
#   status       text not null default 'queued',
#   frame_count  int,
#   video_url    text,
#   error        text,
#   created_at   timestamptz default now(),
#   updated_at   timestamptz default now()
# );
#
# ──────────────────────────────────────────────────────────────────────────────


async def create_job(job_id: str, frame_count: int):
    db = get_client()
    db.table("jobs").insert({
        "id":          job_id,
        "status":      "queued",
        "frame_count": frame_count,
        "created_at":  datetime.utcnow().isoformat(),
        "updated_at":  datetime.utcnow().isoformat(),
    }).execute()


async def update_job(job_id: str, **kwargs):
    db = get_client()
    db.table("jobs").update({
        **kwargs,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()


async def get_job(job_id: str) -> Optional[dict]:
    db = get_client()
    result = db.table("jobs").select("*").eq("id", job_id).maybe_single().execute()
    return result.data if result else None


async def upload_video(job_id: str, file_path: str) -> str:
    """Upload MP4 to Supabase Storage and return public URL."""
    db = get_client()
    storage_path = f"videos/{job_id}.mp4"

    with open(file_path, "rb") as f:
        db.storage.from_("launchvid-outputs").upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "video/mp4"},
        )

    public_url = db.storage.from_("launchvid-outputs").get_public_url(storage_path)
    return public_url