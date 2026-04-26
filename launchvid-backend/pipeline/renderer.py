import json
import os
import subprocess
from pathlib import Path
from typing import Any


def render_video(storyboard: dict[str, Any], voiceover_path: str, output_dir: str = "outputs") -> str:
    """
    Trigger Remotion render via subprocess.
    Expects `npx remotion` availability in environment.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "final-video.mp4")
    input_payload = {"storyboard": storyboard, "voiceoverPath": voiceover_path}

    command = [
        "npx",
        "remotion",
        "render",
        "src/index.ts",
        "LaunchVidComposition",
        output_path,
        "--props",
        json.dumps(input_payload),
    ]

    # If remotion is not set up yet, allow local API testing without hard fail.
    if os.getenv("SKIP_REMOTION", "false").lower() == "true":
        return output_path

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Remotion render failed: {result.stderr.strip()}")
    return output_path
