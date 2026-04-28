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
    # FIX 7 applied: Exponential backoff retry logic
    frame_name    = frame.get("frameName", "Unknown")
    png_b64       = frame.get("fullPngBase64", "")
    layers        = frame.get("layers", {})
    layer_tree_str = _truncate_layer_tree(layers)

    prompt = FRAME_ANALYSIS_PROMPT.format(layer_tree=layer_tree_str)

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
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

            response = _model.generate_content(contents)
            result   = _parse_json_response(response.text)
            result["frame_name"] = frame_name
            result["frame_id"]   = frame.get("frameId", "")
            result["width"]      = frame.get("width", 0)
            result["height"]     = frame.get("height", 0)

            # FIX 4 applied: Apply semantic rules to animation sequence
            if "animation_sequence" in result:
                result["animation_sequence"] = apply_semantic_rules(result["animation_sequence"])

            return result

        except json.JSONDecodeError as e:
            print(f"[vision] Attempt {attempt + 1}/{max_attempts}: JSON parse error for '{frame_name}': {e}")
            if attempt < max_attempts - 1:
                # Exponential backoff: 1s, 2s, 4s
                delay = 2 ** attempt
                print(f"[vision] Retrying after {delay}s...")
                import asyncio
                await asyncio.sleep(delay)
            else:
                print(f"[vision] All retries failed for '{frame_name}'")
                return _fallback_analysis(frame)

        except Exception as e:
            # Check for rate limit (429) or other errors
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            
            if is_rate_limit:
                print(f"[vision] Attempt {attempt + 1}/{max_attempts}: Rate limit error for '{frame_name}': {e}")
                if attempt < max_attempts - 1:
                    # Longer delay for rate limits
                    delay = 10
                    print(f"[vision] Rate limited, retrying after {delay}s...")
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    print(f"[vision] Rate limit persisted after retries for '{frame_name}'")
                    return _fallback_analysis(frame)
            else:
                print(f"[vision] Attempt {attempt + 1}/{max_attempts}: Gemini error for '{frame_name}': {e}")
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt
                    print(f"[vision] Retrying after {delay}s...")
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    print(f"[vision] All retries failed for '{frame_name}': {e}")
                    return _fallback_analysis(frame)

    # Should never reach here, but as a final fallback
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


# FIX 4 applied: Semantic animation rules to override/refine Gemini suggestions
def apply_semantic_rules(animation_sequence: list[dict], layers_map: dict | None = None) -> list[dict]:
    """
    Post-process animation sequence with deterministic semantic rules.
    
    Args:
        animation_sequence: List of animation steps from Gemini
        layers_map: Optional map of layer_id → LayerData for accessing layer properties
    
    Returns:
        Modified animation sequence with semantic rules applied
    """
    if not animation_sequence:
        return animation_sequence

    result = []

    for anim in animation_sequence:
        name = anim.get("layer_name", "").lower()
        layer_type = anim.get("layer_type", "")
        delay_ms = anim.get("delay_ms", 0)
        duration_ms = anim.get("duration_ms", 400)
        easing = anim.get("easing", "ease_out")

        # Rule 1: Background/rect layers → fade_in quickly
        if ("background" in name or "bg" in name or "rect" in name) and layer_type == "RECTANGLE":
            anim["animation"] = "fade_in"
            anim["delay_ms"] = 0
            anim["duration_ms"] = 300
            result.append(anim)
            continue

        # Rule 2: Large text (title, headline, heading, header) or fontSize > 40 → slide_up with spring
        is_title_like = any(x in name for x in ["title", "headline", "heading", "header"])
        if layer_type == "TEXT" and is_title_like:
            anim["animation"] = "slide_up"
            anim["easing"] = "spring"
            anim["duration_ms"] = 500
            result.append(anim)
            continue

        # Rule 3: Button/CTA layers → scale_in with delayed start
        if any(x in name for x in ["button", "cta", "btn"]):
            anim["animation"] = "scale_in"
            anim["delay_ms"] = delay_ms + 150
            anim["duration_ms"] = 400
            result.append(anim)
            continue

        # Rule 4: Image-filled rectangles → scale_in smoothly
        if layer_type == "RECTANGLE" and anim.get("animation") == "fade_in":
            # Check if this might be an image (would be indicated by exportedImageBase64)
            # For now, we apply this to any fade_in rectangle
            anim["animation"] = "scale_in"
            anim["duration_ms"] = 600
            anim["easing"] = "ease_in_out"
            result.append(anim)
            continue

        # Rule 5: Vector/Boolean operations → fade_in
        if layer_type in ["VECTOR", "BOOLEAN_OPERATION"]:
            anim["animation"] = "fade_in"
            anim["duration_ms"] = 400
            result.append(anim)
            continue

        # Rule 6: Ellipse → scale_in
        if layer_type == "ELLIPSE":
            anim["animation"] = "scale_in"
            anim["duration_ms"] = 300
            result.append(anim)
            continue

        result.append(anim)

    # Rule 7: Detect stagger — if 3+ layers have similar delays and y positions
    # Group by delay (within 50ms tolerance)
    delay_groups: dict[int, list] = {}
    for anim in result:
        delay = anim.get("delay_ms", 0)
        # Find group this delay belongs to (within 50ms)
        found_group = None
        for group_delay in delay_groups:
            if abs(delay - group_delay) <= 50:
                found_group = group_delay
                break

        if found_group is None:
            found_group = delay
            delay_groups[found_group] = []

        delay_groups[found_group].append(anim)

    # Apply stagger to groups with 3+ items
    for group_delay, group_anims in delay_groups.items():
        if len(group_anims) >= 3:
            # Sort by current delay, then apply stagger
            for i, anim in enumerate(sorted(group_anims, key=lambda x: x.get("delay_ms", 0))):
                anim["delay_ms"] = group_delay + (i * 80)

    return result


async def analyze_all_frames(frames: list[dict]) -> list[dict]:
    """Analyze all frames sequentially (Gemini free tier: 15 RPM)."""
    results = []
    for i, frame in enumerate(frames):
        print(f"[vision] Analyzing frame {i+1}/{len(frames)}: {frame.get('frameName')}")
        result = await analyze_frame(frame)
        results.append(result)
    return results