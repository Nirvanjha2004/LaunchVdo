"""
pipeline/renderer.py
────────────────────
Calls the Remotion renderer (Node.js project) via subprocess.
Remotion takes the full render props JSON and outputs an MP4.

The Remotion project lives in REMOTION_DIR (see .env).
Run `npm install` there once before using this.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
import shutil

from config import OUTPUT_DIR, REMOTION_DIR


async def render_video(
    job_id: str,
    frames: list[dict],
    frames_analysis: list[dict],
    storyboard: dict,
    scenes_with_audio: list[dict],
) -> str:
    """
    Trigger Remotion render and return the output MP4 path.

    Args:
        job_id:             unique job identifier
        frames:             original plugin export (layer trees + PNGs)
        frames_analysis:    per-frame animation plans from vision.py
        storyboard:         full storyboard from storyboard.py
        scenes_with_audio:  scenes enriched with audio_path from tts.py

    Returns:
        Absolute path to the rendered MP4 file
    """
    # FIX 1: output_dir ko start mein hi absolute path bana diya
    output_dir = os.path.abspath(os.path.join(OUTPUT_DIR, job_id))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "launch_video.mp4")

    # Build the full props object that Remotion will receive
    render_props = _build_render_props(
        job_id, frames, frames_analysis, storyboard, scenes_with_audio
    )

    # Write props to a temp JSON file (avoid shell arg length limits)
    props_path = os.path.join(output_dir, "render_props.json")
    with open(props_path, "w") as f:
        json.dump(render_props, f)

    print(f"[renderer] Starting Remotion render for job {job_id}")
    print(f"[renderer] Output: {output_path}")

    remotion_dir = os.path.abspath(REMOTION_DIR)
    if not os.path.exists(remotion_dir):
        raise FileNotFoundError(
            f"Remotion directory not found: {remotion_dir}\n"
            "Set REMOTION_DIR in .env to point to your Remotion project."
        )

    npx_path = shutil.which("npx")
    if not npx_path:
        raise FileNotFoundError(
            "npx not found in PATH. Make sure Node.js is installed: https://nodejs.org"
        )

    cmd = [
        npx_path,               # full path like C:\...\npx.cmd
        "remotion", "render",
        "LaunchVideo",
        output_path,
        f"--props={props_path}",  # FIX 2: Remotion ka alternative API use kiya hai
        "--log", "verbose",
    ]

    # Run Remotion in a subprocess — it's a long-running Node.js process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=remotion_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},   # ← pass full environment so Node can find itself
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()[-2000:]  # last 2000 chars of error
        print(f"[renderer] Remotion failed:\n{error_msg}")
        raise RuntimeError(f"Remotion render failed: {error_msg}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Remotion finished but MP4 not found at {output_path}")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[renderer] Render complete: {output_path} ({size_mb:.1f} MB)")
    return output_path


def _build_render_props(
    job_id: str,
    frames: list[dict],
    frames_analysis: list[dict],
    storyboard: dict,
    scenes_with_audio: list[dict],
) -> dict:
    """
    Build the full props object that Remotion receives.
    This is the single source of truth for the renderer.
    """

    # Build a lookup: frameId → analysis
    analysis_map = {a["frame_id"]: a for a in frames_analysis}

    # Enrich each frame with its analysis
    enriched_frames = []
    for frame in frames:
        frame_id = frame.get("frameId", "")
        analysis = analysis_map.get(frame_id, {})
        enriched_frames.append({
            # Core frame data
            "frameId":        frame_id,
            "frameName":      frame.get("frameName", ""),
            "width":          frame.get("width", 390),
            "height":         frame.get("height", 844),
            "fullPngBase64":  frame.get("fullPngBase64", ""),
            "layers":         frame.get("layers", {}),
            # Animation plan from Gemini
            "screenPurpose":      analysis.get("screen_purpose", "feature"),
            "headline":           analysis.get("headline", ""),
            "bgColor":            analysis.get("bg_color", "#ffffff"),
            "animationSequence":  analysis.get("animation_sequence", []),
        })

    # Enrich scenes with audio paths
    scenes_clean = []
    for scene in scenes_with_audio:
        audio_path = scene.get("audio_path")
        scenes_clean.append({
            "sceneIndex":      scene["scene_index"],
            "sceneType":       scene.get("scene_type", "screen"),
            "startFrame":      scene["start_frame"],
            "durationFrames":  scene["duration_frames"],
            "transitionIn":    scene.get("transition_in", "fade"),
            "transitionOut":   scene.get("transition_out", "fade"),
            "screenIndex":     scene.get("screen_index"),
            "narration":       scene.get("narration", ""),
            "audioPath":       audio_path,
        })

    return {
        "jobId":       job_id,
        "appName":     storyboard.get("app_name", "App"),
        "tagline":     storyboard.get("tagline", ""),
        "totalFrames": storyboard.get("total_frames", 900),
        "fps":         storyboard.get("fps", 30),
        "scenes":      scenes_clean,
        "frames":      enriched_frames,
    }