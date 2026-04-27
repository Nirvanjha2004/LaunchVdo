"""
pipeline/vision.py
──────────────────
Sends each exported Figma frame (PNG + layer tree) to Gemini 2.5 Flash.
Returns a structured animation plan for each frame.
"""

import base64
import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY

# ── Configure Gemini ───────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-2.5-flash")

# ── Prompt ─────────────────────────────────────────────────────────────────────

FRAME_ANALYSIS_PROMPT = """
You are helping build an animated app launch video (like Apple App Store previews).
Analyze this UI screen and return an animation plan.

Layer tree (JSON):
{layer_tree}

Rules:
- "narration" must be ≤ 15 words, present-tense, exciting
- animation_sequence must cover ALL meaningful visible layers
- delays are cumulative: background=0, hero image=100, headline=200, body=350, CTA=500
- valid animations: fade_in, slide_up, slide_right, slide_left, scale_in, pulse, typewriter
- "pulse" is only for CTA buttons
- "typewriter" is only for TEXT nodes with short text (< 30 chars)

Return ONLY valid JSON, no markdown fences, no explanation:
{{
  "screen_purpose": "onboarding|dashboard|feature|landing|checkout|profile|settings",
  "headline": "the most prominent text on screen (exact characters)",
  "narration": "voiceover sentence for this screen",
  "bg_color": "#hexcode",
  "animation_sequence": [
    {{
      "layer_id": "node id string",
      "layer_name": "node name",
      "layer_type": "TEXT|RECTANGLE|VECTOR|ELLIPSE|FRAME|BOOLEAN_OPERATION",
      "animation": "fade_in|slide_up|slide_right|slide_left|scale_in|pulse|typewriter",
      "delay_ms": 0,
      "duration_ms": 400,
      "easing": "ease_out|ease_in_out|spring"
    }}
  ]
}}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _truncate_layer_tree(layers: dict, max_chars: int = 4000) -> str:
    """
    Truncate the layer tree JSON to stay within Gemini's context window.
    Strips exportedImageBase64 and exportedSvg from child nodes (too large)
    but keeps them at the top level for reference.
    """
    def strip_blobs(node: dict, depth: int = 0) -> dict:
        cleaned = {k: v for k, v in node.items()
                   if k not in ("exportedImageBase64", "exportedSvg", "fullPngBase64")}
        if "children" in node and depth < 3:
            cleaned["children"] = [strip_blobs(c, depth + 1) for c in node.get("children", [])]
        else:
            cleaned.pop("children", None)
        return cleaned

    stripped = strip_blobs(layers)
    serialized = json.dumps(stripped, indent=2)
    return serialized[:max_chars] + ("\n... (truncated)" if len(serialized) > max_chars else "")


def _parse_json_response(text: str) -> dict:
    """Safely parse JSON from Gemini — handles accidental markdown fences."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Main function ──────────────────────────────────────────────────────────────

async def analyze_frame(frame: dict) -> dict:
    """
    Analyze one exported Figma frame.

    Args:
        frame: one item from the plugin's "frames" array — must contain
               fullPngBase64 (str) and layers (dict)

    Returns:
        dict with keys: screen_purpose, headline, narration, bg_color,
                        animation_sequence
    """
    frame_name    = frame.get("frameName", "Unknown")
    png_b64       = frame.get("fullPngBase64", "")
    layers        = frame.get("layers", {})
    layer_tree_str = _truncate_layer_tree(layers)

    prompt = FRAME_ANALYSIS_PROMPT.format(layer_tree=layer_tree_str)

    contents = [prompt]

    # Attach the PNG if available
    if png_b64:
        try:
            image_bytes = base64.b64decode(png_b64)
            contents = [
                {"mime_type": "image/png", "data": image_bytes},
                prompt,
            ]
        except Exception as e:
            print(f"[vision] Could not decode PNG for frame '{frame_name}': {e}")

    try:
        response = _model.generate_content(contents)
        result   = _parse_json_response(response.text)
        result["frame_name"] = frame_name
        result["frame_id"]   = frame.get("frameId", "")
        result["width"]      = frame.get("width", 0)
        result["height"]     = frame.get("height", 0)
        return result

    except json.JSONDecodeError as e:
        print(f"[vision] JSON parse error for '{frame_name}': {e}")
        print(f"[vision] Raw response: {response.text[:500]}")
        # Return a safe fallback so the pipeline doesn't crash
        return _fallback_analysis(frame)

    except Exception as e:
        print(f"[vision] Gemini error for '{frame_name}': {e}")
        return _fallback_analysis(frame)


def _fallback_analysis(frame: dict) -> dict:
    """Minimal fallback when Gemini fails — animates everything with a simple fade."""
    layers   = frame.get("layers", {})
    children = layers.get("children", [])

    sequence = []
    for i, child in enumerate(children[:10]):
        sequence.append({
            "layer_id":   child.get("id", ""),
            "layer_name": child.get("name", ""),
            "layer_type": child.get("type", "RECTANGLE"),
            "animation":  "fade_in",
            "delay_ms":   i * 100,
            "duration_ms": 400,
            "easing":     "ease_out",
        })

    return {
        "frame_name":       frame.get("frameName", "Unknown"),
        "frame_id":         frame.get("frameId", ""),
        "width":            frame.get("width", 390),
        "height":           frame.get("height", 844),
        "screen_purpose":   "feature",
        "headline":         "",
        "narration":        "Introducing a powerful new experience.",
        "bg_color":         "#ffffff",
        "animation_sequence": sequence,
    }


async def analyze_all_frames(frames: list[dict]) -> list[dict]:
    """Analyze all frames sequentially (Gemini free tier: 15 RPM)."""
    results = []
    for i, frame in enumerate(frames):
        print(f"[vision] Analyzing frame {i+1}/{len(frames)}: {frame.get('frameName')}")
        result = await analyze_frame(frame)
        results.append(result)
    return results