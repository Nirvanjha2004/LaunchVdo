"""
pipeline/vision.py
──────────────────
Sends each exported Figma frame (PNG + layer tree) to Gemini 2.5 Flash.
Returns a structured animation plan for each frame.
"""

import asyncio
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
You are a precision UI analysis engine helping build an animated app launch video (like Apple App Store previews).
Your primary task is to extract EXACT pixel-level element data from this Figma frame export.

Layer tree (JSON):
{layer_tree}

## CRITICAL POSITIONING RULES
- Report ALL coordinates as integers in pixels, measured from the TOP-LEFT corner of the frame
- x: horizontal distance from left edge of frame to left edge of element (integer px)
- y: vertical distance from top edge of frame to top edge of element (integer px)
- width: element width in pixels (integer, must be > 0)
- height: element height in pixels (integer, must be > 0)
- Accuracy requirement: ±1px tolerance — read coordinates directly from the layer tree data
- z_index: layer stacking order (0 = bottom/background, higher = closer to viewer)
  - Assign z_index based on visual stacking: background layers get 0, overlapping elements get higher values
  - Elements that visually appear on top of others must have a higher z_index

## ELEMENT CLASSIFICATION RULES
Classify each element as exactly one of these types:
- "text": Any TEXT node (labels, headings, paragraphs, captions)
- "image": RECTANGLE or FRAME with an exported image fill or photo content
- "button": Interactive element with label text, typically rounded rectangle with fill
- "container": FRAME or GROUP that holds other elements (layout containers, cards, modals)
- "icon": Small VECTOR, BOOLEAN_OPERATION, or ELLIPSE used as a symbol/glyph (< 64px in both dimensions)
- "background": Full-width/full-height RECTANGLE or FRAME at the bottom of the stack
- "shape": Decorative RECTANGLE, ELLIPSE, VECTOR, or BOOLEAN_OPERATION that is not an icon
- "divider": Thin horizontal or vertical line element (height ≤ 2px or width ≤ 2px)

## SEMANTIC ROLE RULES
Assign exactly one semantic role per element:
- "hero_image": The largest, most visually prominent image or illustration on screen
- "headline": The largest or most prominent text element (primary heading)
- "subheadline": Secondary heading or supporting text directly below the headline
- "body_text": Paragraph or descriptive text content
- "cta_button": The primary call-to-action button (e.g., "Get Started", "Sign Up", "Buy Now")
- "secondary_button": Secondary or ghost buttons
- "navigation": Tab bars, nav bars, back buttons, menu icons
- "background": Background layer (solid color, gradient, or image behind all content)
- "decoration": Purely decorative shapes, lines, or abstract elements
- "app_icon": App logo or icon mark
- "feature_image": Screenshot, mockup, or feature illustration (not the hero)
- "input_field": Text input, search bar, or form field
- "card": A contained card or list item component
- "status_bar": iOS/Android status bar area
- "label": Small descriptive label or tag

## VISUAL PROPERTIES RULES
- color: Dominant fill color as "#RRGGBB" hex. For text, use the text color. For images, use the dominant color.
  Use "#000000" if unknown or transparent.
- opacity: Value from 0.0 (invisible) to 1.0 (fully opaque). Default to 1.0 if not specified.
- font_size: Integer pixel size for TEXT elements only. Set to null for non-text elements.
- font_weight: "thin|light|regular|medium|semibold|bold|extrabold|black" for TEXT elements. null for non-text.
- text_content: Exact text string for TEXT elements (copy verbatim). null for non-text elements.
- border_radius: Corner radius in pixels (integer). 0 if no rounding. null if not applicable.

## ANIMATION RULES
- animation_sequence must cover ALL meaningful visible layers (skip layers with opacity=0 or visibility=false)
- delays are cumulative from scene start: background=0ms, hero=100ms, headline=200ms, body=350ms, CTA=500ms
- valid animations: fade_in, slide_up, slide_right, slide_left, scale_in, pulse, typewriter, zoom_in, elastic_bounce
- "pulse" is ONLY for CTA buttons
- "typewriter" is ONLY for TEXT nodes with text_content shorter than 30 characters
- "elastic_bounce" is for hero images and prominent feature images
- "zoom_in" is for app icons and small decorative elements

## NARRATION RULES
- "narration" must be ≤ 15 words, present-tense, exciting, describing what the screen does

Return ONLY valid JSON (no markdown fences, no explanation, no trailing commas):
{{
  "screen_purpose": "onboarding|dashboard|feature|landing|checkout|profile|settings",
  "headline": "the most prominent text on screen (exact characters, empty string if none)",
  "narration": "voiceover sentence for this screen",
  "bg_color": "#hexcode of the frame background",
  "frame_width": 390,
  "frame_height": 844,
  "elements": [
    {{
      "layer_id": "node id string from layer tree",
      "layer_name": "node name from layer tree",
      "layer_type": "TEXT|RECTANGLE|VECTOR|ELLIPSE|FRAME|BOOLEAN_OPERATION|GROUP|COMPONENT",
      "element_type": "text|image|button|container|icon|background|shape|divider",
      "semantic_role": "hero_image|headline|subheadline|body_text|cta_button|secondary_button|navigation|background|decoration|app_icon|feature_image|input_field|card|status_bar|label",
      "x": 0,
      "y": 0,
      "width": 100,
      "height": 50,
      "z_index": 0,
      "color": "#hexcode",
      "opacity": 1.0,
      "font_size": null,
      "font_weight": null,
      "text_content": null,
      "border_radius": null
    }}
  ],
  "animation_sequence": [
    {{
      "layer_id": "node id string",
      "layer_name": "node name",
      "layer_type": "TEXT|RECTANGLE|VECTOR|ELLIPSE|FRAME|BOOLEAN_OPERATION",
      "animation": "fade_in|slide_up|slide_right|slide_left|scale_in|pulse|typewriter|zoom_in|elastic_bounce",
      "delay_ms": 0,
      "duration_ms": 400,
      "easing": "ease_out|ease_in_out|spring"
    }}
  ]
}}
"""

# ── Coordinate Extraction Utilities ───────────────────────────────────────────

def extract_precise_coordinates(layer: dict, frame_width: int, frame_height: int) -> dict:
    """
    Extract pixel-perfect integer coordinates from a Figma layer dict.

    Figma layer trees carry x, y, width, height as the ground truth.  This
    function:
      1. Reads raw values from the layer (floats are allowed by Figma).
      2. Rounds floating-point values to the nearest integer (±1px accuracy).
      3. Clamps coordinates so the element stays within the frame bounds.
      4. Handles device-pixel-ratio (DPR) scaling when the layer carries a
         ``devicePixelRatio`` field (or the frame carries one at the top level).

    Args:
        layer:        A single Figma layer dict (may contain x, y, width,
                      height, devicePixelRatio).
        frame_width:  Width of the containing frame in logical pixels.
        frame_height: Height of the containing frame in logical pixels.

    Returns:
        dict with integer keys: x, y, width, height — all guaranteed to be
        non-negative integers within the frame bounds.
    """
    # --- 1. Read raw values (default to safe fallbacks) ----------------------
    raw_x      = layer.get("x", 0) or 0
    raw_y      = layer.get("y", 0) or 0
    raw_width  = layer.get("width", 0) or 0
    raw_height = layer.get("height", 0) or 0

    # --- 2. Apply DPR scaling if present -------------------------------------
    dpr = layer.get("devicePixelRatio", 1) or 1
    if dpr != 1 and dpr > 0:
        raw_x      = raw_x      / dpr
        raw_y      = raw_y      / dpr
        raw_width  = raw_width  / dpr
        raw_height = raw_height / dpr

    # --- 3. Round to nearest integer (±1px accuracy) -------------------------
    x      = int(round(raw_x))
    y      = int(round(raw_y))
    width  = int(round(raw_width))
    height = int(round(raw_height))

    # --- 4. Ensure positive dimensions ---------------------------------------
    width  = max(1, width)
    height = max(1, height)

    # --- 5. Clamp to frame bounds --------------------------------------------
    # x must be in [0, frame_width - 1]; width must not exceed frame right edge
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width  = min(width,  frame_width  - x)
    height = min(height, frame_height - y)

    # Final safety: dimensions must be at least 1px after clamping
    width  = max(1, width)
    height = max(1, height)

    return {"x": x, "y": y, "width": width, "height": height}


def validate_element_coordinates(element: dict, frame_width: int, frame_height: int) -> dict:
    """
    Validate and correct coordinates in an element dict (as returned by Gemini
    or built by ``_fallback_analysis``).

    Reads x, y, width, height from *element*, applies the same rounding and
    clamping logic as ``extract_precise_coordinates``, and returns a **new**
    dict with corrected integer values merged in.  All other keys are
    preserved unchanged.

    Args:
        element:      Element dict (must have x, y, width, height keys).
        frame_width:  Width of the containing frame in logical pixels.
        frame_height: Height of the containing frame in logical pixels.

    Returns:
        A copy of *element* with x, y, width, height replaced by validated
        integer values within frame bounds.
    """
    corrected = extract_precise_coordinates(element, frame_width, frame_height)
    return {**element, **corrected}


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


def _index_layers(layer: dict) -> dict[str, dict]:
    """Flatten the layer tree into a lookup by layer id."""
    lookup: dict[str, dict] = {}

    def walk(node: dict) -> None:
        node_id = node.get("id")
        if node_id:
            lookup[node_id] = node
        for child in node.get("children", []) or []:
            walk(child)

    walk(layer or {})
    return lookup


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
    layer_lookup = _index_layers(layers)

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

            # Enrich elements array with layer metadata (exported images, etc.)
            # Also validate/correct coordinates returned by Gemini
            gemini_frame_width  = result.get("frame_width",  frame.get("width",  390))
            gemini_frame_height = result.get("frame_height", frame.get("height", 844))
            for element in result.get("elements", []):
                layer_meta = layer_lookup.get(element.get("layer_id", ""), {})
                element["__exported_image"] = layer_meta.get("exportedImageBase64")
                # Prefer ground-truth coordinates from the layer tree when available
                if layer_meta:
                    ground_truth = extract_precise_coordinates(
                        layer_meta, gemini_frame_width, gemini_frame_height
                    )
                    element.update(ground_truth)
                else:
                    # Validate/clamp whatever Gemini returned
                    corrected = validate_element_coordinates(
                        element, gemini_frame_width, gemini_frame_height
                    )
                    element.update({
                        "x": corrected["x"],
                        "y": corrected["y"],
                        "width": corrected["width"],
                        "height": corrected["height"],
                    })

            for anim in result.get("animation_sequence", []):
                layer_meta = layer_lookup.get(anim.get("layer_id", ""), {})
                anim["__font_size"] = layer_meta.get("text", {}).get("fontSize")
                anim["__exported_image"] = layer_meta.get("exportedImageBase64")
                anim["__y"] = layer_meta.get("y")
                anim["__height"] = layer_meta.get("height")

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
                    await asyncio.sleep(delay)
                else:
                    print(f"[vision] Rate limit persisted after retries for '{frame_name}'")
                    return _fallback_analysis(frame)
            else:
                print(f"[vision] Attempt {attempt + 1}/{max_attempts}: Gemini error for '{frame_name}': {e}")
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt
                    print(f"[vision] Retrying after {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"[vision] All retries failed for '{frame_name}': {e}")
                    return _fallback_analysis(frame)

    # Should never reach here, but as a final fallback
    return _fallback_analysis(frame)


def _fallback_analysis(frame: dict) -> dict:
    """Minimal fallback when Gemini fails — animates everything with a simple fade."""
    layers       = frame.get("layers", {})
    children     = layers.get("children", [])
    frame_width  = frame.get("width", 390)
    frame_height = frame.get("height", 844)

    elements = []
    sequence = []
    for i, child in enumerate(children[:10]):
        layer_id   = child.get("id", "")
        layer_name = child.get("name", "")
        layer_type = child.get("type", "RECTANGLE")
        x      = child.get("x", 0)
        y      = child.get("y", 0)
        width  = child.get("width", frame_width)
        height = child.get("height", 50)

        # Infer basic element_type and semantic_role from layer type and position
        if layer_type == "TEXT":
            element_type  = "text"
            semantic_role = "headline" if i == 0 else "body_text"
        elif i == 0 and width >= frame_width * 0.9:
            element_type  = "background"
            semantic_role = "background"
        else:
            element_type  = "shape"
            semantic_role = "decoration"

        raw_element = {
            "layer_id":      layer_id,
            "layer_name":    layer_name,
            "layer_type":    layer_type,
            "element_type":  element_type,
            "semantic_role": semantic_role,
            "x":             x,
            "y":             y,
            "width":         width,
            "height":        height,
            "z_index":       i,
            "color":         "#000000",
            "opacity":       1.0,
            "font_size":     None,
            "font_weight":   None,
            "text_content":  None,
            "border_radius": None,
        }
        # Validate and clamp coordinates to frame bounds
        validated = validate_element_coordinates(raw_element, frame_width, frame_height)
        elements.append(validated)

        sequence.append({
            "layer_id":   layer_id,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "animation":  "fade_in",
            "delay_ms":   i * 100,
            "duration_ms": 400,
            "easing":     "ease_out",
        })

    return {
        "frame_name":         frame.get("frameName", "Unknown"),
        "frame_id":           frame.get("frameId", ""),
        "width":              frame_width,
        "height":             frame_height,
        "screen_purpose":     "feature",
        "headline":           "",
        "narration":          "Introducing a powerful new experience.",
        "bg_color":           "#ffffff",
        "elements":           elements,
        "animation_sequence": sequence,
    }


# FIX 4 applied: Semantic animation rules to override/refine Gemini suggestions
def apply_semantic_rules(animation_sequence: list[dict]) -> list[dict]:
    """
    Post-process animation sequence with deterministic semantic rules.
    
    Args:
        animation_sequence: List of animation steps from Gemini
    
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
        font_size = anim.get("__font_size")
        exported_image = anim.get("__exported_image")

        if ("background" in name or "bg" in name or "rect" in name) and layer_type == "RECTANGLE":
            anim["animation"] = "fade_in"
            anim["delay_ms"] = 0
            anim["duration_ms"] = 300
            anim["easing"] = anim.get("easing", "ease_out")
        elif layer_type == "TEXT" and (
            any(x in name for x in ["title", "headline", "heading", "header"]) or (font_size is not None and font_size > 40)
        ):
            anim["animation"] = "slide_up"
            anim["duration_ms"] = 500
            anim["easing"] = "spring"
        elif any(x in name for x in ["button", "cta", "btn"]):
            anim["animation"] = "scale_in"
            anim["delay_ms"] = delay_ms + 150
            anim["duration_ms"] = 400
        elif layer_type == "RECTANGLE" and exported_image:
            anim["animation"] = "scale_in"
            anim["duration_ms"] = 600
            anim["easing"] = "ease_in_out"
        elif layer_type in ["VECTOR", "BOOLEAN_OPERATION"]:
            anim["animation"] = "fade_in"
            anim["duration_ms"] = 400
        elif layer_type == "ELLIPSE":
            anim["animation"] = "scale_in"
            anim["duration_ms"] = 300

        result.append(anim)

    # Rule 7: Detect stagger — if 3+ layers have delay within 50ms and similar y positions
    groups: list[list[dict]] = []
    for anim in result:
        delay = anim.get("delay_ms", 0)
        y = anim.get("__y")
        height = anim.get("__height") or 0

        matched_group = None
        for group in groups:
            anchor = group[0]
            anchor_delay = anchor.get("delay_ms", 0)
            anchor_y = anchor.get("__y")
            anchor_height = anchor.get("__height") or 0
            y_threshold = max(24, min(anchor_height, height) * 0.15 if anchor_height and height else 24)
            if abs(delay - anchor_delay) <= 50 and y is not None and anchor_y is not None and abs(y - anchor_y) <= y_threshold:
                matched_group = group
                break

        if matched_group is None:
            groups.append([anim])
        else:
            matched_group.append(anim)

    for group in groups:
        if len(group) >= 3:
            sorted_group = sorted(group, key=lambda item: (item.get("__y", 0), item.get("delay_ms", 0)))
            base_delay = min(item.get("delay_ms", 0) for item in sorted_group)
            for index, anim in enumerate(sorted_group):
                anim["delay_ms"] = base_delay + (index * 80)

    for anim in result:
        anim.pop("__font_size", None)
        anim.pop("__exported_image", None)
        anim.pop("__y", None)
        anim.pop("__height", None)

    return result


async def analyze_all_frames(frames: list[dict]) -> list[dict]:
    """Analyze all frames sequentially (Gemini free tier: 15 RPM)."""
    results = []
    for i, frame in enumerate(frames):
        print(f"[vision] Analyzing frame {i+1}/{len(frames)}: {frame.get('frameName')}")
        result = await analyze_frame(frame)
        results.append(result)
    return results