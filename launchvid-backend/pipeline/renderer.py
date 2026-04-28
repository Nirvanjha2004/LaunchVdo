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
) -> dict:
    """
    FIX 9 applied: Render three output formats.
    
    Trigger Remotion render for three formats and return their paths.

    Args:
        job_id:             unique job identifier
        frames:             original plugin export (layer trees + PNGs)
        frames_analysis:    per-frame animation plans from vision.py
        storyboard:         full storyboard from storyboard.py
        scenes_with_audio:  scenes enriched with audio_path from tts.py

    Returns:
        dict with keys: portrait_path, landscape_path, square_path
    """
    output_dir = os.path.abspath(os.path.join(OUTPUT_DIR, job_id))
    os.makedirs(output_dir, exist_ok=True)

    # Build the full props object that Remotion will receive
    render_props = _build_render_props(
        job_id, frames, frames_analysis, storyboard, scenes_with_audio
    )

    # Write props to a temp JSON file (avoid shell arg length limits)
    props_path = os.path.join(output_dir, "render_props.json")
    with open(props_path, "w") as f:
        json.dump(render_props, f)

    # FIX 9 applied: Render three formats
    formats = [
        {"id": "LaunchVideoPortrait", "suffix": "_portrait", "width": 1080, "height": 1920},
        {"id": "LaunchVideoLandscape", "suffix": "_landscape", "width": 1920, "height": 1080},
        {"id": "LaunchVideoSquare", "suffix": "_square", "width": 1080, "height": 1080},
    ]

    render_paths = {}

    for fmt in formats:
        try:
            output_path = os.path.join(output_dir, f"launch_video{fmt['suffix']}.mp4")
            print(f"[renderer] Starting {fmt['id']} render for job {job_id}")
            print(f"[renderer] Output: {output_path} ({fmt['width']}x{fmt['height']})")

            await _run_remotion_render(
                job_id=job_id,
                composition_id=fmt["id"],
                output_path=output_path,
                props_path=props_path,
                width=fmt["width"],
                height=fmt["height"],
            )

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Remotion finished but MP4 not found at {output_path}")

            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"[renderer] Render complete: {output_path} ({size_mb:.1f} MB)")
            render_paths[fmt["suffix"]] = output_path

        except Exception as e:
            print(f"[renderer] Error rendering {fmt['id']}: {e}")
            raise

    return render_paths


async def _run_remotion_render(
    job_id: str,
    composition_id: str,
    output_path: str,
    props_path: str,
    width: int,
    height: int,
):
    """
    Execute a single Remotion render command.
    """
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
        npx_path,
        "remotion", "render",
        composition_id,
        output_path,
        f"--props={props_path}",
        f"--width={width}",
        f"--height={height}",
        "--log", "verbose",
    ]

    # Run Remotion in a subprocess
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=remotion_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()[-2000:]
        print(f"[renderer] Remotion failed:\n{error_msg}")
        raise RuntimeError(f"Remotion render failed: {error_msg}")


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
        
        # FIX 3: Clean path for Remotion Web URL format
        if audio_path:
            # 1. Backslashes (\) ko forward slashes (/) mein badlein
            audio_path = audio_path.replace("\\", "/")
            
            # 2. Extra prefix hata dein taaki Remotion ke public folder se direct resolve ho
            if "launchvid-remotion/public/" in audio_path:
                audio_path = audio_path.split("launchvid-remotion/public/")[-1]
            elif "public/" in audio_path:
                audio_path = audio_path.split("public/")[-1]

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