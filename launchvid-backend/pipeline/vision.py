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


# ── Element Bounds Calculation ────────────────────────────────────────────────

def calculate_element_bounds(
    layer: dict,
    frame_width: int,
    frame_height: int,
    export_scale: float = 1.0,
) -> dict:
    """
    Calculate an element's bounding box in the frame's coordinate space,
    handling Figma's ``absoluteBoundingBox`` vs relative x/y, export scale
    factors, and normalized bounds.

    Figma layers can carry two kinds of positional data:

    * **Relative x/y** – coordinates relative to the parent frame/group.
    * **absoluteBoundingBox** – coordinates in the global Figma canvas space.

    When ``absoluteBoundingBox`` is present the function converts it to
    frame-relative coordinates by subtracting the frame's own canvas origin
    (taken from the layer's ``frameOrigin`` field, or defaulting to ``(0, 0)``
    when not provided).

    The export scale factor accounts for Figma exports at different
    resolutions (e.g. 2× exports double all pixel values).  Dividing by
    ``export_scale`` converts back to logical (1×) pixel coordinates.

    Args:
        layer:        A Figma layer dict.  May contain any combination of:
                      ``x``, ``y``, ``width``, ``height`` (relative coords),
                      ``absoluteBoundingBox`` (dict with ``x``, ``y``,
                      ``width``, ``height`` in canvas space),
                      ``frameOrigin`` (dict with ``x``, ``y`` giving the
                      frame's canvas origin, used to convert absolute coords
                      to frame-relative coords).
        frame_width:  Width of the containing frame in logical pixels (after
                      any scale correction).
        frame_height: Height of the containing frame in logical pixels.
        export_scale: Scale factor used when the frame was exported (e.g.
                      ``2.0`` for a 2× export).  All raw pixel values are
                      divided by this factor.  Must be > 0; defaults to 1.0.

    Returns:
        dict with the following keys:

        * ``x`` (int)      – frame-relative left edge in logical pixels.
        * ``y`` (int)      – frame-relative top edge in logical pixels.
        * ``width`` (int)  – element width in logical pixels (≥ 1).
        * ``height`` (int) – element height in logical pixels (≥ 1).
        * ``bounds_normalized`` (dict) – normalised bounds relative to the
          frame dimensions, each value a float in [0.0, 1.0]:

          * ``left``   – x / frame_width
          * ``top``    – y / frame_height
          * ``right``  – (x + width) / frame_width
          * ``bottom`` – (y + height) / frame_height
    """
    # --- 0. Sanitise scale factor -------------------------------------------
    if not export_scale or export_scale <= 0:
        export_scale = 1.0

    # --- 1. Prefer absoluteBoundingBox when available -----------------------
    abs_bb = layer.get("absoluteBoundingBox")
    if abs_bb and isinstance(abs_bb, dict):
        # Canvas-space coordinates
        canvas_x      = abs_bb.get("x", 0) or 0
        canvas_y      = abs_bb.get("y", 0) or 0
        raw_width     = abs_bb.get("width",  0) or 0
        raw_height    = abs_bb.get("height", 0) or 0

        # Frame's own canvas origin (so we can make coords frame-relative)
        frame_origin  = layer.get("frameOrigin") or {}
        origin_x      = frame_origin.get("x", 0) or 0
        origin_y      = frame_origin.get("y", 0) or 0

        raw_x = canvas_x - origin_x
        raw_y = canvas_y - origin_y
    else:
        # Fall back to relative x/y
        raw_x      = layer.get("x", 0) or 0
        raw_y      = layer.get("y", 0) or 0
        raw_width  = layer.get("width",  0) or 0
        raw_height = layer.get("height", 0) or 0

    # --- 2. Apply export scale factor ---------------------------------------
    raw_x      = raw_x      / export_scale
    raw_y      = raw_y      / export_scale
    raw_width  = raw_width  / export_scale
    raw_height = raw_height / export_scale

    # --- 3. Round to nearest integer (±1px accuracy) -----------------------
    x      = int(round(raw_x))
    y      = int(round(raw_y))
    width  = int(round(raw_width))
    height = int(round(raw_height))

    # --- 4. Ensure positive dimensions -------------------------------------
    width  = max(1, width)
    height = max(1, height)

    # --- 5. Clamp to frame bounds ------------------------------------------
    x = max(0, min(x, frame_width  - 1))
    y = max(0, min(y, frame_height - 1))
    width  = min(width,  frame_width  - x)
    height = min(height, frame_height - y)

    # Final safety: dimensions must be at least 1px after clamping
    width  = max(1, width)
    height = max(1, height)

    # --- 6. Compute normalised bounds (0.0 – 1.0) --------------------------
    # Guard against zero-dimension frames (should never happen in practice)
    fw = frame_width  if frame_width  > 0 else 1
    fh = frame_height if frame_height > 0 else 1

    bounds_normalized = {
        "left":   x / fw,
        "top":    y / fh,
        "right":  (x + width)  / fw,
        "bottom": (y + height) / fh,
    }

    return {
        "x":                x,
        "y":                y,
        "width":            width,
        "height":           height,
        "bounds_normalized": bounds_normalized,
    }


def calculate_relative_size(
    element: dict,
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Return a normalised size score (0.0 – 1.0) representing how much of the
    frame the element occupies by area.

    The score is the ratio of the element's area to the frame's total area,
    clamped to [0.0, 1.0].  It is useful for visual hierarchy analysis: a
    full-screen background returns 1.0, a small icon returns a value close
    to 0.0.

    Args:
        element:      A dict with at least ``width`` and ``height`` keys
                      (integer or float pixel values).  Missing or non-positive
                      values are treated as 0.
        frame_width:  Width of the containing frame in pixels (must be > 0).
        frame_height: Height of the containing frame in pixels (must be > 0).

    Returns:
        Float in [0.0, 1.0] representing the element's area as a fraction of
        the frame area.  Returns 0.0 when the frame has zero area or the
        element has non-positive dimensions.
    """
    frame_area = frame_width * frame_height
    if frame_area <= 0:
        return 0.0

    elem_width  = element.get("width",  0) or 0
    elem_height = element.get("height", 0) or 0

    if elem_width <= 0 or elem_height <= 0:
        return 0.0

    ratio = (elem_width * elem_height) / frame_area
    return min(1.0, max(0.0, ratio))


# ── Element Overlap Detection and Z-Index Analysis ────────────────────────────

def detect_element_overlaps(elements: list[dict]) -> list[dict]:
    """
    Detect overlapping elements and calculate overlap areas.

    For each pair of elements, computes the intersection rectangle.  If the
    intersection has positive area, an overlap record is emitted.

    The ``overlap_ratio`` is the overlap area divided by the area of the
    *smaller* element (the one with the lesser area).  This gives a value in
    [0.0, 1.0] that represents how much of the smaller element is covered.

    Args:
        elements: List of element dicts, each containing at minimum:
                  ``layer_id`` (str), ``x`` (int), ``y`` (int),
                  ``width`` (int), ``height`` (int).

    Returns:
        List of overlap records, each a dict with:
        - ``element_a_id`` (str) – layer_id of the first element.
        - ``element_b_id`` (str) – layer_id of the second element.
        - ``overlap_area``  (int) – pixel area of the intersection rectangle.
        - ``overlap_ratio`` (float) – overlap_area / area of the smaller element.
    """
    overlaps: list[dict] = []

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            a = elements[i]
            b = elements[j]

            # Compute intersection rectangle
            ax1, ay1 = a.get("x", 0), a.get("y", 0)
            ax2 = ax1 + max(a.get("width", 0), 0)
            ay2 = ay1 + max(a.get("height", 0), 0)

            bx1, by1 = b.get("x", 0), b.get("y", 0)
            bx2 = bx1 + max(b.get("width", 0), 0)
            by2 = by1 + max(b.get("height", 0), 0)

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1

            if inter_w <= 0 or inter_h <= 0:
                continue  # no overlap

            overlap_area = int(inter_w * inter_h)

            area_a = max(a.get("width", 0), 0) * max(a.get("height", 0), 0)
            area_b = max(b.get("width", 0), 0) * max(b.get("height", 0), 0)
            smaller_area = min(area_a, area_b)

            overlap_ratio = (overlap_area / smaller_area) if smaller_area > 0 else 0.0

            overlaps.append({
                "element_a_id": a.get("layer_id", ""),
                "element_b_id": b.get("layer_id", ""),
                "overlap_area": overlap_area,
                "overlap_ratio": overlap_ratio,
            })

    return overlaps


def assign_z_indices(layers: list[dict]) -> list[dict]:
    """
    Assign z-index values to a flat list of layers produced by
    ``collect_layers_depth_first``.

    The z-index reflects the visual stacking order: elements that appear
    *later* in DFS traversal order (higher ``z_order``) and elements that are
    *deeper* in the tree (higher ``depth``) should appear above their
    predecessors.

    Formula::

        z_index = z_order * (max_depth + 1) + depth

    This ensures:
    - Children always have a higher z-index than their parent.
    - Later siblings always have a higher z-index than earlier siblings.

    Args:
        layers: Flat list of layer dicts as returned by
                ``collect_layers_depth_first``.  Each dict must have
                ``z_order`` (int) and ``depth`` (int) fields.

    Returns:
        A new list of layer dicts (shallow copies) with a ``z_index`` (int)
        field added to each.  The input list is not mutated.
    """
    if not layers:
        return []

    max_depth = max((layer.get("depth", 0) for layer in layers), default=0)
    depth_factor = max_depth + 1  # multiplier so depth never overflows into z_order

    result = []
    for layer in layers:
        z_order = layer.get("z_order", 0)
        depth = layer.get("depth", 0)
        z_index = z_order * depth_factor + depth
        result.append({**layer, "z_index": z_index})

    return result


def analyze_visual_stack(elements: list[dict]) -> list[dict]:
    """
    Enrich each element with overlap information derived from
    ``detect_element_overlaps``.

    Each element in the returned list gains two new fields:

    - ``overlaps_with`` (list[str]) – layer_ids of elements that overlap with
      this element.
    - ``overlap_count`` (int) – number of elements this element overlaps with.

    Args:
        elements: List of element dicts from Gemini analysis.  Each dict must
                  have a ``layer_id`` key and valid ``x``, ``y``, ``width``,
                  ``height`` values.

    Returns:
        A new list of element dicts (shallow copies) enriched with
        ``overlaps_with`` and ``overlap_count``.  The input list is not
        mutated.
    """
    overlap_records = detect_element_overlaps(elements)

    # Build a mapping: layer_id → set of overlapping layer_ids
    overlap_map: dict[str, set[str]] = {
        elem.get("layer_id", ""): set() for elem in elements
    }

    for record in overlap_records:
        id_a = record["element_a_id"]
        id_b = record["element_b_id"]
        if id_a in overlap_map:
            overlap_map[id_a].add(id_b)
        if id_b in overlap_map:
            overlap_map[id_b].add(id_a)

    result = []
    for element in elements:
        layer_id = element.get("layer_id", "")
        overlapping_ids = sorted(overlap_map.get(layer_id, set()))
        result.append({
            **element,
            "overlaps_with": overlapping_ids,
            "overlap_count": len(overlapping_ids),
        })

    return result


# ── Layer Tree Traversal ───────────────────────────────────────────────────────

def collect_layers_depth_first(root: dict, max_depth: int = 10) -> list[dict]:
    """
    Traverse a Figma layer tree using depth-first search (pre-order DFS) and
    return a flat list of all visible layers enriched with traversal metadata.

    The traversal:
      - Visits each node *before* its children (pre-order: parent first).
      - Skips invisible layers (``visible == False``).
      - Handles circular references safely by tracking visited node IDs.
      - Respects ``max_depth`` to prevent infinite recursion on deeply nested
        trees (FR1.7: up to 100 layers within 10 seconds per frame).

    Each collected layer dict is a **shallow copy** of the original node
    (without its ``children`` key) enriched with two extra fields:

    - ``depth``   – nesting level of the node (root = 0).
    - ``z_order`` – sequential 0-based integer reflecting DFS traversal order.

    Args:
        root:      Root node of the Figma layer tree (a dict with optional
                   ``children`` list, ``id``, ``visible``, etc.).
        max_depth: Maximum nesting depth to traverse.  Nodes deeper than this
                   are silently skipped.  Defaults to 10.

    Returns:
        Flat list of layer dicts in DFS pre-order, each enriched with
        ``depth`` and ``z_order``.  Invisible layers and nodes beyond
        ``max_depth`` are excluded.
    """
    result: list[dict] = []
    visited: set[str] = set()

    def _dfs(node: dict, depth: int) -> None:
        # Respect max_depth limit
        if depth > max_depth:
            return

        # Skip invisible layers
        if node.get("visible") is False:
            return

        # Circular reference guard (only for nodes that have an id)
        node_id = node.get("id")
        if node_id is not None:
            if node_id in visited:
                return
            visited.add(node_id)

        # Build enriched copy (exclude children to keep the flat list clean)
        enriched = {k: v for k, v in node.items() if k != "children"}
        enriched["depth"] = depth
        enriched["z_order"] = len(result)  # assigned before appending
        result.append(enriched)

        # Recurse into children (DFS pre-order: parent already added above)
        for child in node.get("children", []) or []:
            _dfs(child, depth + 1)

    if root:
        _dfs(root, depth=0)

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _truncate_layer_tree(layers: dict, max_chars: int = 4000) -> str:
    """
    Build an informative layer summary for the Gemini prompt.

    Uses ``collect_layers_depth_first`` to produce a flat, ordered list of
    visible layers enriched with ``depth`` and ``z_order``.  Each layer entry
    retains its key properties (id, name, type, x, y, width, height, opacity,
    visible) while large binary blobs (exportedImageBase64, exportedSvg,
    fullPngBase64) are stripped to keep the prompt within Gemini's context
    window.

    Falls back to the legacy strip-and-truncate approach when the layer tree
    is empty or ``collect_layers_depth_first`` returns no results.

    Args:
        layers:    Root node of the Figma layer tree.
        max_chars: Maximum character length of the returned string.

    Returns:
        JSON string (possibly truncated) suitable for embedding in the Gemini
        prompt.
    """
    _BLOB_KEYS = frozenset({"exportedImageBase64", "exportedSvg", "fullPngBase64"})

    # --- Attempt DFS-based summary -------------------------------------------
    flat_layers = collect_layers_depth_first(layers, max_depth=10)

    if flat_layers:
        # Build a compact representation: keep key properties + depth/z_order
        _KEEP_KEYS = frozenset({
            "id", "name", "type", "x", "y", "width", "height",
            "opacity", "visible", "depth", "z_order",
            # Preserve text content and fill info for Gemini
            "characters", "fills", "strokes", "cornerRadius",
            "absoluteBoundingBox", "constraints",
        })
        summary = []
        for layer in flat_layers:
            entry = {k: v for k, v in layer.items()
                     if k in _KEEP_KEYS and k not in _BLOB_KEYS}
            summary.append(entry)

        serialized = json.dumps(summary, indent=2)
        return serialized[:max_chars] + ("\n... (truncated)" if len(serialized) > max_chars else "")

    # --- Legacy fallback: strip blobs and truncate ---------------------------
    def strip_blobs(node: dict, depth: int = 0) -> dict:
        cleaned = {k: v for k, v in node.items() if k not in _BLOB_KEYS}
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

            # Task 1.1.5: Enrich elements with overlap detection and z-index analysis
            if "elements" in result:
                # Build a z_index lookup from the DFS layer traversal so that
                # stacking order is grounded in the actual Figma layer tree
                # rather than relying solely on Gemini's z_index estimates.
                flat_layers = collect_layers_depth_first(layers, max_depth=10)
                layers_with_z = assign_z_indices(flat_layers)
                z_index_by_id: dict[str, int] = {
                    layer.get("id", ""): layer["z_index"]
                    for layer in layers_with_z
                    if layer.get("id")
                }

                # Override each element's z_index with the tree-derived value
                # when available; keep Gemini's value as a fallback.
                for element in result["elements"]:
                    lid = element.get("layer_id", "")
                    if lid in z_index_by_id:
                        element["z_index"] = z_index_by_id[lid]

                # Enrich with pairwise overlap data
                result["elements"] = analyze_visual_stack(result["elements"])

            # Task 1.1.6: Detect element groups and enrich elements with group membership
            if "elements" in result:
                groups = detect_element_groups(result["elements"])
                result["element_groups"] = groups
                result["elements"] = _apply_group_dependencies(result["elements"], groups)

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

    # Task 1.1.5: Assign z-indices from DFS traversal, then enrich with overlap data
    flat_layers = collect_layers_depth_first(layers, max_depth=10)
    layers_with_z = assign_z_indices(flat_layers)
    z_index_by_id: dict[str, int] = {
        layer.get("id", ""): layer["z_index"]
        for layer in layers_with_z
        if layer.get("id")
    }
    for element in elements:
        lid = element.get("layer_id", "")
        if lid in z_index_by_id:
            element["z_index"] = z_index_by_id[lid]

    elements = analyze_visual_stack(elements)

    # Task 1.1.6: Detect element groups and enrich elements with group membership
    groups = detect_element_groups(elements)
    elements = _apply_group_dependencies(elements, groups)

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
        "element_groups":     groups,
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


# ── Element Grouping Detection ────────────────────────────────────────────────

# Semantic roles that are compatible with each group type.
# Each entry maps a group_type to the set of semantic_role values that can
# participate in that group.
_GROUP_ROLE_COMPATIBILITY: dict[str, frozenset[str]] = {
    "hero_section":  frozenset({"headline", "subheadline", "body_text", "cta_button", "hero_image"}),
    "card":          frozenset({"headline", "subheadline", "body_text", "cta_button", "secondary_button",
                                "feature_image", "card", "label", "decoration"}),
    "form_group":    frozenset({"label", "input_field", "cta_button", "secondary_button", "body_text"}),
    "nav_group":     frozenset({"navigation", "app_icon", "label", "decoration"}),
    "feature_row":   frozenset({"headline", "subheadline", "body_text", "decoration", "feature_image",
                                "hero_image", "label"}),
    "cta_group":     frozenset({"cta_button", "secondary_button", "body_text", "label", "decoration"}),
    "content_block": frozenset({"headline", "subheadline", "body_text", "label", "decoration",
                                "feature_image", "card"}),
}

# Minimum number of members required to form a group.
_MIN_GROUP_SIZE = 2

# Spatial proximity threshold in pixels: elements within this distance of each
# other (edge-to-edge) are considered spatially proximate.
_PROXIMITY_PX = 20


def _elements_are_proximate(a: dict, b: dict, threshold_px: int = _PROXIMITY_PX) -> bool:
    """
    Return True when two elements are spatially proximate or overlapping.

    Two elements are proximate when the gap between their closest edges is
    ≤ *threshold_px* pixels in both the horizontal and vertical axes.  An
    overlap (negative gap) also satisfies this condition.

    Args:
        a, b:         Element dicts with ``x``, ``y``, ``width``, ``height``.
        threshold_px: Maximum edge-to-edge gap (in pixels) to be considered
                      proximate.  Defaults to ``_PROXIMITY_PX`` (20 px).

    Returns:
        bool
    """
    ax1, ay1 = a.get("x", 0), a.get("y", 0)
    ax2 = ax1 + max(a.get("width", 0), 0)
    ay2 = ay1 + max(a.get("height", 0), 0)

    bx1, by1 = b.get("x", 0), b.get("y", 0)
    bx2 = bx1 + max(b.get("width", 0), 0)
    by2 = by1 + max(b.get("height", 0), 0)

    # Horizontal gap: negative means overlap
    h_gap = max(ax1, bx1) - min(ax2, bx2)
    # Vertical gap: negative means overlap
    v_gap = max(ay1, by1) - min(ay2, by2)

    return h_gap <= threshold_px and v_gap <= threshold_px


def _group_bounding_box(members: list[dict]) -> dict:
    """
    Compute the axis-aligned bounding box that encloses all *members*.

    Args:
        members: Non-empty list of element dicts with ``x``, ``y``,
                 ``width``, ``height``.

    Returns:
        dict with ``x``, ``y``, ``width``, ``height`` (all ints).
    """
    min_x = min(m.get("x", 0) for m in members)
    min_y = min(m.get("y", 0) for m in members)
    max_x = max(m.get("x", 0) + max(m.get("width", 0), 0) for m in members)
    max_y = max(m.get("y", 0) + max(m.get("height", 0), 0) for m in members)
    return {
        "x":      int(min_x),
        "y":      int(min_y),
        "width":  int(max(1, max_x - min_x)),
        "height": int(max(1, max_y - min_y)),
    }


def _classify_group(member_roles: list[str]) -> tuple[str, float]:
    """
    Determine the best-matching group type and a confidence score for a set
    of semantic roles.

    The confidence score is the Jaccard-like ratio of matched roles to the
    union of the candidate group's role set and the member roles:

        confidence = |matched| / |member_roles|

    where *matched* is the number of member roles that appear in the
    candidate group's compatible-role set.

    Ties in score are broken by a specificity ranking so that more specific
    group types (e.g. ``cta_group``, ``form_group``) win over generic ones
    (e.g. ``card``, ``hero_section``) when the scores are equal.

    Args:
        member_roles: List of ``semantic_role`` strings for the candidate
                      group members.

    Returns:
        Tuple of (group_type: str, confidence: float).  Returns
        ``("content_block", 0.0)`` as a safe fallback when no specific
        pattern matches.
    """
    if not member_roles:
        return ("content_block", 0.0)

    role_set = set(member_roles)
    best_type = "content_block"
    best_score = 0.0

    # Specificity priority: used as a tie-breaker only when a group's defining
    # "key signal" role is present.  A group without its key signal role does
    # not earn a specificity advantage.
    # Higher number = higher priority when scores are equal AND key role present.
    _SPECIFICITY: dict[str, int] = {
        "form_group":    6,  # key signal: input_field
        "nav_group":     5,  # key signal: navigation
        "cta_group":     4,  # key signal: cta_button
        "hero_section":  3,  # key signal: headline or hero_image
        "card":          2,
        "feature_row":   1,
        "content_block": 0,  # fallback
    }

    # Mapping from group_type to its defining "key signal" roles.
    _KEY_SIGNALS: dict[str, frozenset[str]] = {
        "form_group":   frozenset({"input_field"}),
        "nav_group":    frozenset({"navigation"}),
        "cta_group":    frozenset({"cta_button"}),
        "hero_section": frozenset({"headline", "hero_image"}),
    }

    for group_type, compatible_roles in _GROUP_ROLE_COMPATIBILITY.items():
        matched = len(role_set & compatible_roles)
        if matched == 0:
            continue
        # Score: fraction of member roles that are compatible with this group type
        score = matched / len(role_set)
        # Bonus for specific high-signal roles
        if group_type == "hero_section" and (
            "headline" in role_set or "hero_image" in role_set
        ):
            score = min(1.0, score + 0.2)
        elif group_type == "form_group" and "input_field" in role_set:
            score = min(1.0, score + 0.2)
        elif group_type == "nav_group" and "navigation" in role_set:
            score = min(1.0, score + 0.2)
        elif group_type == "cta_group" and "cta_button" in role_set:
            score = min(1.0, score + 0.15)

        # Tie-breaking: prefer the more specific group type, but only when
        # that group's key signal role is present in the member set.
        has_key_signal = bool(role_set & _KEY_SIGNALS.get(group_type, frozenset()))
        current_specificity = _SPECIFICITY.get(group_type, 0) if has_key_signal else 0
        best_has_key = bool(role_set & _KEY_SIGNALS.get(best_type, frozenset()))
        best_specificity = _SPECIFICITY.get(best_type, 0) if best_has_key else 0
        if score > best_score or (score == best_score and current_specificity > best_specificity):
            best_score = score
            best_type = group_type

    # Minimum confidence floor so callers can filter low-quality groups
    if best_score < 0.3:
        best_score = 0.3

    return (best_type, round(best_score, 4))


def detect_element_groups(
    elements: list[dict],
    proximity_px: int = _PROXIMITY_PX,
) -> list[dict]:
    """
    Detect groups of related UI elements using spatial proximity and semantic
    role compatibility.

    The algorithm uses a union-find (disjoint-set) approach:

    1. For every pair of elements, check whether they are spatially proximate
       (edge-to-edge gap ≤ *proximity_px*) **and** share at least one
       compatible group type (i.e. both roles appear in the same entry of
       ``_GROUP_ROLE_COMPATIBILITY``).
    2. Merge proximate, semantically compatible elements into the same group.
    3. Discard groups with fewer than ``_MIN_GROUP_SIZE`` members.
    4. For each surviving group, classify it, compute its bounding box, and
       assign a confidence score.

    Args:
        elements:     List of element dicts.  Each dict must have at minimum:
                      ``layer_id`` (str), ``x`` (int), ``y`` (int),
                      ``width`` (int), ``height`` (int), and
                      ``semantic_role`` (str).
        proximity_px: Maximum edge-to-edge pixel gap for two elements to be
                      considered spatially proximate.  Defaults to 20 px.

    Returns:
        List of group dicts, each containing:

        - ``group_id``    (str)       – unique identifier, e.g. ``"group_0"``.
        - ``group_type``  (str)       – detected pattern type (e.g.
          ``"hero_section"``, ``"card"``, ``"form_group"``).
        - ``member_ids``  (list[str]) – ``layer_id`` values of member elements,
          in the order they appear in *elements*.
        - ``bounding_box`` (dict)     – ``{x, y, width, height}`` enclosing all
          members (integer pixels).
        - ``confidence``  (float)     – score in [0.0, 1.0] reflecting how well
          the members match the detected group type.
    """
    if not elements:
        return []

    n = len(elements)

    # ── Union-Find helpers ────────────────────────────────────────────────────
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    # ── Build a set of roles that each element participates in ────────────────
    # Pre-compute which group types each element's role is compatible with.
    def _compatible_group_types(role: str) -> frozenset[str]:
        return frozenset(
            gt for gt, roles in _GROUP_ROLE_COMPATIBILITY.items() if role in roles
        )

    elem_group_types = [
        _compatible_group_types(elem.get("semantic_role", "")) for elem in elements
    ]

    # ── Pairwise proximity + semantic compatibility check ─────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if already in the same group
            if _find(i) == _find(j):
                continue

            # Must share at least one compatible group type
            shared_types = elem_group_types[i] & elem_group_types[j]
            if not shared_types:
                continue

            # Must be spatially proximate
            if _elements_are_proximate(elements[i], elements[j], proximity_px):
                _union(i, j)

    # ── Collect groups ────────────────────────────────────────────────────────
    from collections import defaultdict
    groups_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups_map[_find(idx)].append(idx)

    result: list[dict] = []
    group_counter = 0

    for root_idx, member_indices in sorted(groups_map.items()):
        if len(member_indices) < _MIN_GROUP_SIZE:
            continue

        member_elements = [elements[i] for i in member_indices]
        member_ids = [elem.get("layer_id", "") for elem in member_elements]
        member_roles = [elem.get("semantic_role", "") for elem in member_elements]

        group_type, confidence = _classify_group(member_roles)
        bbox = _group_bounding_box(member_elements)

        result.append({
            "group_id":    f"group_{group_counter}",
            "group_type":  group_type,
            "member_ids":  member_ids,
            "bounding_box": bbox,
            "confidence":  confidence,
        })
        group_counter += 1

    return result


def _apply_group_dependencies(
    elements: list[dict],
    groups: list[dict],
) -> list[dict]:
    """
    Enrich each element with a ``group_id`` and a ``dependencies`` list that
    references the other members of its group.

    Elements that do not belong to any group receive an empty ``dependencies``
    list and ``group_id`` of ``None``.

    Args:
        elements: List of element dicts (each must have ``layer_id``).
        groups:   List of group dicts as returned by ``detect_element_groups``.

    Returns:
        A new list of element dicts (shallow copies) with ``group_id`` and
        ``dependencies`` fields added.  The input list is not mutated.
    """
    # Build a mapping: layer_id → (group_id, [other member layer_ids])
    membership: dict[str, tuple[str, list[str]]] = {}
    for group in groups:
        gid = group["group_id"]
        members = group["member_ids"]
        for lid in members:
            others = [m for m in members if m != lid]
            membership[lid] = (gid, others)

    result = []
    for element in elements:
        lid = element.get("layer_id", "")
        if lid in membership:
            gid, deps = membership[lid]
            result.append({**element, "group_id": gid, "dependencies": deps})
        else:
            result.append({**element, "group_id": None, "dependencies": []})

    return result


async def analyze_all_frames(frames: list[dict]) -> list[dict]:
    """Analyze all frames sequentially (Gemini free tier: 15 RPM)."""
    results = []
    for i, frame in enumerate(frames):
        print(f"[vision] Analyzing frame {i+1}/{len(frames)}: {frame.get('frameName')}")
        result = await analyze_frame(frame)
        results.append(result)
    return results