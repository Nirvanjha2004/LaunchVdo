"""
pipeline/storyboard.py
──────────────────────
Takes the per-frame analysis from vision.py and calls Groq to produce
a full 30-second video storyboard with scene timings, transitions,
and a cohesive narrative.
"""

import json
import re
from groq import Groq
from config import GROQ_API_KEY

# ── Client ─────────────────────────────────────────────────────────────────────

_client = Groq(api_key=GROQ_API_KEY)

# ── Prompt ─────────────────────────────────────────────────────────────────────

STORYBOARD_PROMPT = """
You are creating a 30-second animated app launch video storyboard.

App description: {app_description}

Analyzed screens:
{frames_analysis}

Rules:
- Total duration = exactly 30 seconds = 900 frames at 30fps
- Each scene gets between 120–300 frames (4–10 seconds)
- The intro card (app name + tagline) always comes first: 60 frames (2 seconds)
- Outro card (CTA) always comes last: 60 frames (2 seconds)
- Distribute remaining frames proportionally across screens
- valid transition_in values: slide_up, slide_right, fade, zoom_in, slide_left
- valid transition_out values: slide_up, fade, zoom_out, slide_left
- The narration across all scenes should tell a cohesive story

Return ONLY valid JSON, no markdown fences:
{{
  "app_name": "name of the app (infer from screens if not provided)",
  "tagline": "one punchy line, max 8 words",
  "total_frames": 900,
  "fps": 30,
  "scenes": [
    {{
      "scene_index": 0,
      "scene_type": "intro_card",
      "start_frame": 0,
      "duration_frames": 60,
      "transition_in": "fade",
      "transition_out": "slide_up",
      "screen_index": null,
      "narration": "Introducing the app"
    }},
    {{
      "scene_index": 1,
      "scene_type": "screen",
      "start_frame": 60,
      "duration_frames": 195,
      "transition_in": "slide_up",
      "transition_out": "slide_left",
      "screen_index": 0,
      "narration": "narration for this screen from analysis"
    }}
  ]
}}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _slim_analysis(frames_analysis: list[dict]) -> list[dict]:
    """Strip large fields before sending to Groq."""
    slim = []
    for f in frames_analysis:
        slim.append({
            "frame_index":    frames_analysis.index(f),
            "frame_name":     f.get("frame_name", ""),
            "screen_purpose": f.get("screen_purpose", "feature"),
            "headline":       f.get("headline", ""),
            "narration":      f.get("narration", ""),
            "bg_color":       f.get("bg_color", "#ffffff"),
            "layer_count":    len(f.get("animation_sequence", [])),
        })
    return slim


# ── Main function ──────────────────────────────────────────────────────────────

async def generate_storyboard(
    frames_analysis: list[dict],
    app_description: str = "",
) -> dict:
    """
    Generate a full video storyboard from analyzed frames.

    Args:
        frames_analysis: list of dicts returned by vision.analyze_all_frames()
        app_description: optional string from user (can be empty)

    Returns:
        dict with keys: app_name, tagline, total_frames, fps, scenes
    """
    slim = _slim_analysis(frames_analysis)
    prompt = STORYBOARD_PROMPT.format(
        app_description=app_description or "a beautiful mobile app",
        frames_analysis=json.dumps(slim, indent=2),
    )

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000,
        )
        result = _parse_json(response.choices[0].message.content)
        print(f"[storyboard] Generated {len(result.get('scenes', []))} scenes")
        return result

    except json.JSONDecodeError as e:
        print(f"[storyboard] JSON parse error: {e}")
        return _fallback_storyboard(frames_analysis)

    except Exception as e:
        print(f"[storyboard] Groq error: {e}")
        return _fallback_storyboard(frames_analysis)


def _fallback_storyboard(frames_analysis: list[dict]) -> dict:
    """Simple fallback: equal duration per screen, basic transitions."""
    n_screens  = len(frames_analysis)
    total      = 900
    intro_dur  = 60
    outro_dur  = 60
    remaining  = total - intro_dur - outro_dur
    per_screen = remaining // max(n_screens, 1)

    scenes = [
        {
            "scene_index":    0,
            "scene_type":     "intro_card",
            "start_frame":    0,
            "duration_frames": intro_dur,
            "transition_in":  "fade",
            "transition_out": "slide_up",
            "screen_index":   None,
            "narration":      "Introducing a brand new experience.",
        }
    ]

    frame_cursor = intro_dur
    for i, f in enumerate(frames_analysis):
        scenes.append({
            "scene_index":    i + 1,
            "scene_type":     "screen",
            "start_frame":    frame_cursor,
            "duration_frames": per_screen,
            "transition_in":  "slide_up",
            "transition_out": "fade",
            "screen_index":   i,
            "narration":      f.get("narration", "Explore this feature."),
        })
        frame_cursor += per_screen

    scenes.append({
        "scene_index":    len(scenes),
        "scene_type":     "outro_card",
        "start_frame":    frame_cursor,
        "duration_frames": outro_dur,
        "transition_in":  "fade",
        "transition_out": "fade",
        "screen_index":   None,
        "narration":      "Download now.",
    })

    return {
        "app_name":     "App",
        "tagline":      "Built for you.",
        "total_frames": total,
        "fps":          30,
        "scenes":       scenes,
    }