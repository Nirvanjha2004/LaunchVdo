import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from db.supabase import update_job_status
from pipeline.renderer import render_video
from pipeline.storyboard import generate_storyboard
from pipeline.tts import create_voiceover
from pipeline.vision import analyze_frames

load_dotenv()

app = FastAPI(title="LaunchVid Backend", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Video idea/prompt")
    frames: list[str] = Field(default_factory=list, description="Optional frame paths")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate_video(payload: GenerateRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())

    try:
        update_job_status(job_id, "queued", {"prompt": payload.prompt})
        update_job_status(job_id, "analyzing_frames")
        vision_notes = analyze_frames(payload.frames, payload.prompt)

        update_job_status(job_id, "generating_storyboard")
        storyboard = generate_storyboard(payload.prompt, vision_notes)

        update_job_status(job_id, "generating_voiceover")
        voiceover_path = create_voiceover(storyboard.get("voiceover", ""))

        update_job_status(job_id, "rendering_video")
        output_path = render_video(storyboard, voiceover_path)

        update_job_status(job_id, "completed", {"output_path": output_path})
        return {"job_id": job_id, "status": "completed", "output_path": output_path}
    except Exception as exc:
        update_job_status(job_id, "failed", {"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
