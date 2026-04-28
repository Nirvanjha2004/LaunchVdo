from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, OUTPUT_DIR
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
#   video_urls   jsonb,
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


async def upload_video(job_id: str, video_paths: dict) -> dict:
    """
    FIX 9 applied: Upload multiple video formats to Supabase Storage.
    
    Args:
        job_id: Job identifier
        video_paths: dict with keys 'portrait', 'landscape', 'square' containing file paths
    
    Returns:
        dict with keys 'portrait', 'landscape', 'square' containing public URLs
    """
    db = get_client()
    public_urls = {}

    # Upload each format
    for suffix, file_path in video_paths.items():
        if not file_path:
            continue

        storage_path = f"videos/{job_id}_{suffix}.mp4"

        try:
            with open(file_path, "rb") as f:
                db.storage.from_("launchvid-outputs").upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "video/mp4"},
                )

            public_url = db.storage.from_("launchvid-outputs").get_public_url(storage_path)
            public_urls[suffix] = public_url
            print(f"[supabase] Uploaded {suffix} video: {storage_path}")
        except Exception as e:
            print(f"[supabase] Failed to upload {suffix} video: {e}")
            raise

    # FIX 8 applied: Clean up temp files after successful uploads
    try:
        import os
        import shutil

        for suffix, file_path in video_paths.items():
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"[supabase] Deleted {suffix} MP4: {file_path}")

        # Delete the audio directory
        audio_dir = os.path.join(OUTPUT_DIR, job_id, "audio")
        if os.path.exists(audio_dir):
            shutil.rmtree(audio_dir)
            print(f"[supabase] Deleted audio directory: {audio_dir}")

        # Delete render_props.json
        render_props_path = os.path.join(OUTPUT_DIR, job_id, "render_props.json")
        if os.path.exists(render_props_path):
            os.remove(render_props_path)
            print(f"[supabase] Deleted render_props.json: {render_props_path}")

        print(f"[supabase] Cleanup complete for job {job_id}")
    except Exception as e:
        print(f"[supabase] Warning: Cleanup failed for job {job_id}: {e}")
        # Don't fail the upload if cleanup fails

    return public_urls