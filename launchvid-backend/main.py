"""
main.py
───────
FastAPI application for LaunchVid backend.

Routes:
  POST /analyze      → accepts plugin export JSON, kicks off pipeline
  GET  /job/{id}     → returns job status + video URL when done
  GET  /health       → sanity check

Pipeline (runs in background after POST /analyze):
  1. vision.py      → Gemini 2.5 Flash analyzes each frame
  2. storyboard.py  → Groq builds the video storyboard
  3. tts.py         → gTTS generates voiceover MP3s
  4. renderer.py    → Remotion renders the final MP4
  5. db/supabase.py → uploads MP4, updates job status
"""

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import traceback
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PORT
from db.supabase import create_job, get_job, update_job, upload_video
from pipeline.vision import analyze_all_frames
from pipeline.storyboard import generate_storyboard
from pipeline.tts import generate_all_voiceovers
from pipeline.renderer import render_video
from pipeline.queue import enqueue_job  # FIX 6 applied

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LaunchVid API",
    description="Turns Figma exports into animated launch videos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ──────────────────────────────────────────────────

class ExportPayload(BaseModel):
    frames:         list[dict[str, Any]]
    appDescription: str = ""
    exportedAt:     str = ""


class JobResponse(BaseModel):
    job_id:     str
    status:     str
    video_url:  str | None = None  # Backward compatibility (primary portrait)
    video_urls: dict | None = None  # FIX 9 applied: All three formats
    error:      str | None = None

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "launchvid-api"}


@app.post("/analyze", response_model=JobResponse)
async def analyze(payload: ExportPayload):
    """
    Accept a Figma plugin export and start the video generation pipeline.
    Returns immediately with a job_id. Poll /job/{job_id} for status.
    """
    if not payload.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    job_id = str(uuid.uuid4())

    # Create job record in Supabase
    await create_job(job_id, frame_count=len(payload.frames))
    print(f"[main] Job {job_id} created — {len(payload.frames)} frames")

    # FIX 6 applied: Queue jobs so only one render runs at a time
    asyncio.create_task(enqueue_job(job_id, payload.frames, payload.appDescription))

    print(f"[main] Job {job_id} queued — {len(payload.frames)} frames")
    return JobResponse(job_id=job_id, status="queued")


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Poll this endpoint to check job progress."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=      job_id,
        status=      job["status"],
        video_url=   job.get("video_url"),
        video_urls=  job.get("video_urls"),  # FIX 9 applied
        error=       job.get("error"),
    )

# ── Pipeline orchestrator ──────────────────────────────────────────────────────

async def run_pipeline(job_id: str, frames: list[dict], app_description: str):
    """
    Full pipeline — runs in background after POST /analyze.

    Stages:
      queued → analyzing → storyboarding → generating_audio → rendering → done
                                                                        → failed
    """
    try:
        # ── Stage 1: Vision ──────────────────────────────────────────────────
        await update_job(job_id, status="analyzing")
        print(f"[pipeline:{job_id}] Stage 1/4 — Analyzing {len(frames)} frames with Gemini")
        frames_analysis = await analyze_all_frames(frames)

        # ── Stage 2: Storyboard ──────────────────────────────────────────────
        await update_job(job_id, status="storyboarding")
        print(f"[pipeline:{job_id}] Stage 2/4 — Generating storyboard with Groq")
        storyboard = await generate_storyboard(frames_analysis, app_description)

        # ── Stage 3: TTS ─────────────────────────────────────────────────────
        await update_job(job_id, status="generating_audio")
        print(f"[pipeline:{job_id}] Stage 3/4 — Generating voiceovers")
        scenes_with_audio = await generate_all_voiceovers(storyboard["scenes"], job_id)

        # ── Stage 4: Remotion render ─────────────────────────────────────────
        await update_job(job_id, status="rendering")
        print(f"[pipeline:{job_id}] Stage 4/4 — Rendering video with Remotion")
        render_paths = await render_video(  # FIX 9 applied: Returns dict with three formats
            job_id=job_id,
            frames=frames,
            frames_analysis=frames_analysis,
            storyboard=storyboard,
            scenes_with_audio=scenes_with_audio,
        )

        # ── Upload + finish ───────────────────────────────────────────────────
        print(f"[pipeline:{job_id}] Uploading to Supabase Storage")
        video_urls = await upload_video(job_id, render_paths)  # FIX 9 applied: Returns dict

        # FIX 9 applied: Store all three URLs, use portrait as primary
        primary_url = video_urls.get("portrait", "")
        await update_job(
            job_id,
            status="done",
            video_url=primary_url,  # For backward compatibility
            video_urls=video_urls,   # All three formats
        )
        print(f"[pipeline:{job_id}] ✅ Done — {primary_url}")

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[pipeline:{job_id}] ❌ Failed:\n{error_detail}")
        await update_job(job_id, status="failed", error=str(e))

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)