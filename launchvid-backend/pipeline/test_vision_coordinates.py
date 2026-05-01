"""
Unit tests for extract_precise_coordinates(), validate_element_coordinates(),
and collect_layers_depth_first() in pipeline/vision.py.

Run with:
    python -m pytest pipeline/test_vision_coordinates.py -v
"""

import pytest
import sys
import os

# Allow running from the launchvid-backend directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Patch config so we don't need a real API key during tests
import unittest.mock as mock
with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
    with mock.patch("google.generativeai.configure"):
        with mock.patch("google.generativeai.GenerativeModel"):
            from pipeline.vision import (
                extract_precise_coordinates,
                validate_element_coordinates,
                collect_layers_depth_first,
                _truncate_layer_tree,
                calculate_element_bounds,
                calculate_relative_size,
                detect_element_overlaps,
                assign_z_indices,
                analyze_visual_stack,
            )


# ── extract_precise_coordinates ───────────────────────────────────────────────

class TestExtractPreciseCoordinates:
    """Tests for extract_precise_coordinates()."""

    FRAME_W = 390
    FRAME_H = 844

    def test_integer_passthrough(self):
        """Integer values within bounds are returned unchanged."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result == {"x": 10, "y": 20, "width": 100, "height": 50}

    def test_float_rounding(self):
        """Float values are rounded to the nearest integer (±1px accuracy).

        Python 3 uses banker's rounding (round-half-to-even):
          round(100.5) → 100  (rounds to nearest even)
          round(49.5)  → 50   (rounds to nearest even)
        The key requirement is that the result is within ±1px of the true value.
        """
        layer = {"x": 10.4, "y": 20.6, "width": 100.5, "height": 49.5}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 10
        assert result["y"] == 21
        # round(100.5) == 100 in Python 3 (banker's rounding); still within ±1px
        assert abs(result["width"] - 100.5) <= 1
        # round(49.5) == 50 in Python 3
        assert result["height"] == 50

    def test_float_rounding_small(self):
        """Small float offsets stay within ±1px."""
        layer = {"x": 5.1, "y": 10.9, "width": 80.0, "height": 40.0}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 5
        assert result["y"] == 11
        assert result["width"] == 80
        assert result["height"] == 40

    def test_negative_x_clamped_to_zero(self):
        """Negative x is clamped to 0."""
        layer = {"x": -5, "y": 10, "width": 100, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0

    def test_negative_y_clamped_to_zero(self):
        """Negative y is clamped to 0."""
        layer = {"x": 10, "y": -20, "width": 100, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["y"] == 0

    def test_x_beyond_frame_clamped(self):
        """x beyond frame_width - 1 is clamped."""
        layer = {"x": 400, "y": 10, "width": 50, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == self.FRAME_W - 1

    def test_y_beyond_frame_clamped(self):
        """y beyond frame_height - 1 is clamped."""
        layer = {"x": 10, "y": 900, "width": 50, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["y"] == self.FRAME_H - 1

    def test_width_clamped_to_frame_right_edge(self):
        """Width is clamped so x + width does not exceed frame_width."""
        layer = {"x": 300, "y": 10, "width": 200, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] + result["width"] <= self.FRAME_W

    def test_height_clamped_to_frame_bottom_edge(self):
        """Height is clamped so y + height does not exceed frame_height."""
        layer = {"x": 10, "y": 800, "width": 50, "height": 200}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["y"] + result["height"] <= self.FRAME_H

    def test_zero_width_becomes_one(self):
        """Zero width is corrected to 1 (minimum valid dimension)."""
        layer = {"x": 10, "y": 10, "width": 0, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["width"] >= 1

    def test_zero_height_becomes_one(self):
        """Zero height is corrected to 1."""
        layer = {"x": 10, "y": 10, "width": 50, "height": 0}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["height"] >= 1

    def test_negative_width_becomes_one(self):
        """Negative width is corrected to 1."""
        layer = {"x": 10, "y": 10, "width": -30, "height": 50}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["width"] >= 1

    def test_missing_keys_default_to_zero(self):
        """Missing x/y/width/height keys default to 0 (width/height become 1)."""
        result = extract_precise_coordinates({}, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] >= 1
        assert result["height"] >= 1

    def test_dpr_scaling(self):
        """Coordinates are divided by devicePixelRatio before rounding."""
        # A layer at 2× DPR: raw coords are doubled
        layer = {"x": 20, "y": 40, "width": 200, "height": 100, "devicePixelRatio": 2}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["width"] == 100
        assert result["height"] == 50

    def test_dpr_1_no_change(self):
        """DPR of 1 (default) does not alter coordinates."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50, "devicePixelRatio": 1}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result == {"x": 10, "y": 20, "width": 100, "height": 50}

    def test_dpr_zero_treated_as_one(self):
        """DPR of 0 (invalid) is treated as 1 to avoid division by zero."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50, "devicePixelRatio": 0}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result == {"x": 10, "y": 20, "width": 100, "height": 50}

    def test_full_frame_background(self):
        """A full-frame background layer is returned as-is (no clamping needed)."""
        layer = {"x": 0, "y": 0, "width": self.FRAME_W, "height": self.FRAME_H}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] == self.FRAME_W
        assert result["height"] == self.FRAME_H

    def test_return_type_is_int(self):
        """All returned values are Python ints, not floats."""
        layer = {"x": 1.7, "y": 2.3, "width": 50.9, "height": 30.1}
        result = extract_precise_coordinates(layer, self.FRAME_W, self.FRAME_H)
        for key in ("x", "y", "width", "height"):
            assert isinstance(result[key], int), f"{key} should be int, got {type(result[key])}"


# ── validate_element_coordinates ─────────────────────────────────────────────

class TestValidateElementCoordinates:
    """Tests for validate_element_coordinates()."""

    FRAME_W = 390
    FRAME_H = 844

    def test_preserves_other_keys(self):
        """Non-coordinate keys are preserved unchanged."""
        element = {
            "layer_id": "abc",
            "layer_name": "Button",
            "element_type": "button",
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 50,
            "color": "#ff0000",
            "opacity": 0.9,
        }
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["layer_id"] == "abc"
        assert result["layer_name"] == "Button"
        assert result["element_type"] == "button"
        assert result["color"] == "#ff0000"
        assert result["opacity"] == 0.9

    def test_corrects_out_of_bounds_coordinates(self):
        """Out-of-bounds coordinates are clamped."""
        element = {"x": -10, "y": -5, "width": 500, "height": 1000}
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["x"] >= 0
        assert result["y"] >= 0
        assert result["x"] + result["width"] <= self.FRAME_W
        assert result["y"] + result["height"] <= self.FRAME_H

    def test_does_not_mutate_original(self):
        """The original element dict is not mutated."""
        element = {"x": -5, "y": 10, "width": 100, "height": 50}
        original_x = element["x"]
        validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert element["x"] == original_x  # original unchanged

    def test_float_coordinates_rounded(self):
        """Float coordinates in element dict are rounded to integers."""
        element = {"x": 10.7, "y": 20.2, "width": 100.5, "height": 50.4}
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        for key in ("x", "y", "width", "height"):
            assert isinstance(result[key], int)

    def test_valid_coordinates_unchanged(self):
        """Valid integer coordinates within bounds are returned unchanged."""
        element = {"x": 50, "y": 100, "width": 200, "height": 300}
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 50
        assert result["y"] == 100
        assert result["width"] == 200
        assert result["height"] == 300

    def test_gemini_oversize_width_corrected(self):
        """Gemini sometimes returns width larger than the frame — should be clamped."""
        element = {
            "layer_id": "bg",
            "x": 0,
            "y": 0,
            "width": 9999,
            "height": 9999,
        }
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["width"] <= self.FRAME_W
        assert result["height"] <= self.FRAME_H

    def test_element_at_right_edge(self):
        """Element positioned at the right edge of the frame is handled correctly."""
        element = {"x": self.FRAME_W - 10, "y": 0, "width": 10, "height": 50}
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["x"] + result["width"] <= self.FRAME_W

    def test_element_at_bottom_edge(self):
        """Element positioned at the bottom edge of the frame is handled correctly."""
        element = {"x": 0, "y": self.FRAME_H - 10, "width": 50, "height": 10}
        result = validate_element_coordinates(element, self.FRAME_W, self.FRAME_H)
        assert result["y"] + result["height"] <= self.FRAME_H


# ── collect_layers_depth_first ────────────────────────────────────────────────

class TestCollectLayersDepthFirst:
    """Tests for collect_layers_depth_first()."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_layer(id: str, name: str = "", children: list | None = None,
                    visible: bool = True, **kwargs) -> dict:
        """Build a minimal Figma-like layer dict."""
        node: dict = {"id": id, "name": name or id, "visible": visible}
        node.update(kwargs)
        if children is not None:
            node["children"] = children
        return node

    # ── Basic traversal ───────────────────────────────────────────────────────

    def test_single_root_no_children(self):
        """A root with no children returns a single-element list."""
        root = self._make_layer("root")
        result = collect_layers_depth_first(root)
        assert len(result) == 1
        assert result[0]["id"] == "root"

    def test_dfs_pre_order(self):
        """Parent is collected before its children (pre-order DFS)."""
        root = self._make_layer("A", children=[
            self._make_layer("B", children=[
                self._make_layer("D"),
            ]),
            self._make_layer("C"),
        ])
        result = collect_layers_depth_first(root)
        ids = [n["id"] for n in result]
        assert ids == ["A", "B", "D", "C"]

    def test_z_order_sequential(self):
        """z_order values are sequential integers starting at 0."""
        root = self._make_layer("A", children=[
            self._make_layer("B"),
            self._make_layer("C"),
        ])
        result = collect_layers_depth_first(root)
        z_orders = [n["z_order"] for n in result]
        assert z_orders == list(range(len(result)))

    def test_depth_field_root_is_zero(self):
        """Root node has depth == 0."""
        root = self._make_layer("root")
        result = collect_layers_depth_first(root)
        assert result[0]["depth"] == 0

    def test_depth_field_children(self):
        """Children have depth == 1, grandchildren depth == 2."""
        root = self._make_layer("A", children=[
            self._make_layer("B", children=[
                self._make_layer("C"),
            ]),
        ])
        result = collect_layers_depth_first(root)
        depth_map = {n["id"]: n["depth"] for n in result}
        assert depth_map["A"] == 0
        assert depth_map["B"] == 1
        assert depth_map["C"] == 2

    # ── Invisible layer skipping ──────────────────────────────────────────────

    def test_invisible_root_skipped(self):
        """An invisible root node is not collected."""
        root = self._make_layer("root", visible=False)
        result = collect_layers_depth_first(root)
        assert result == []

    def test_invisible_child_skipped(self):
        """An invisible child (and its subtree) is not collected."""
        root = self._make_layer("A", children=[
            self._make_layer("B", visible=False, children=[
                self._make_layer("C"),  # should also be skipped
            ]),
            self._make_layer("D"),
        ])
        result = collect_layers_depth_first(root)
        ids = [n["id"] for n in result]
        assert "B" not in ids
        assert "C" not in ids
        assert "A" in ids
        assert "D" in ids

    def test_visible_true_is_collected(self):
        """Nodes with visible == True are collected normally."""
        root = self._make_layer("root", visible=True)
        result = collect_layers_depth_first(root)
        assert len(result) == 1

    def test_visible_missing_is_collected(self):
        """Nodes without a 'visible' key are treated as visible."""
        root = {"id": "root", "name": "root"}  # no 'visible' key
        result = collect_layers_depth_first(root)
        assert len(result) == 1

    # ── max_depth ─────────────────────────────────────────────────────────────

    def test_max_depth_zero_collects_only_root(self):
        """max_depth=0 collects only the root node."""
        root = self._make_layer("A", children=[
            self._make_layer("B"),
        ])
        result = collect_layers_depth_first(root, max_depth=0)
        assert len(result) == 1
        assert result[0]["id"] == "A"

    def test_max_depth_one_collects_root_and_direct_children(self):
        """max_depth=1 collects root and its direct children only."""
        root = self._make_layer("A", children=[
            self._make_layer("B", children=[
                self._make_layer("C"),  # depth 2 — should be excluded
            ]),
        ])
        result = collect_layers_depth_first(root, max_depth=1)
        ids = [n["id"] for n in result]
        assert "A" in ids
        assert "B" in ids
        assert "C" not in ids

    def test_max_depth_default_handles_deep_tree(self):
        """Default max_depth=10 handles trees up to 10 levels deep."""
        # Build a chain 10 levels deep
        node = self._make_layer("leaf")
        for i in range(9, -1, -1):
            node = self._make_layer(f"level_{i}", children=[node])
        result = collect_layers_depth_first(node)
        # root (depth 0) + 10 descendants = 11 nodes
        assert len(result) == 11

    def test_max_depth_exceeded_nodes_excluded(self):
        """Nodes beyond max_depth are excluded."""
        # Build a chain 5 levels deep
        node = self._make_layer("level_5")
        for i in range(4, -1, -1):
            node = self._make_layer(f"level_{i}", children=[node])
        result = collect_layers_depth_first(node, max_depth=3)
        depths = [n["depth"] for n in result]
        assert max(depths) <= 3

    # ── Circular reference handling ───────────────────────────────────────────

    def test_circular_reference_does_not_loop(self):
        """Circular references are detected and do not cause infinite loops."""
        child = self._make_layer("child")
        root = self._make_layer("root", children=[child])
        # Manually create a circular reference: child points back to root
        child["children"] = [root]
        result = collect_layers_depth_first(root)
        # Should terminate; root and child each appear exactly once
        ids = [n["id"] for n in result]
        assert ids.count("root") == 1
        assert ids.count("child") == 1

    def test_shared_subtree_visited_once(self):
        """A node referenced from multiple parents is only visited once."""
        shared = self._make_layer("shared")
        root = self._make_layer("root", children=[
            self._make_layer("A", children=[shared]),
            self._make_layer("B", children=[shared]),  # same object
        ])
        result = collect_layers_depth_first(root)
        ids = [n["id"] for n in result]
        assert ids.count("shared") == 1

    # ── Output structure ──────────────────────────────────────────────────────

    def test_children_key_not_in_output(self):
        """The 'children' key is stripped from collected layer dicts."""
        root = self._make_layer("A", children=[self._make_layer("B")])
        result = collect_layers_depth_first(root)
        for node in result:
            assert "children" not in node

    def test_original_properties_preserved(self):
        """Non-traversal properties (name, type, x, y, etc.) are preserved."""
        root = self._make_layer("A", type="FRAME", x=10, y=20, width=100, height=200)
        result = collect_layers_depth_first(root)
        assert result[0]["type"] == "FRAME"
        assert result[0]["x"] == 10
        assert result[0]["y"] == 20
        assert result[0]["width"] == 100
        assert result[0]["height"] == 200

    def test_original_tree_not_mutated(self):
        """The original layer tree is not mutated by the traversal."""
        root = self._make_layer("A", children=[self._make_layer("B")])
        original_children_count = len(root["children"])
        collect_layers_depth_first(root)
        assert len(root["children"]) == original_children_count
        assert "depth" not in root
        assert "z_order" not in root

    def test_empty_root_returns_empty_list(self):
        """An empty dict root returns an empty list."""
        result = collect_layers_depth_first({})
        assert result == []

    def test_none_root_returns_empty_list(self):
        """A None root returns an empty list."""
        result = collect_layers_depth_first(None)  # type: ignore[arg-type]
        assert result == []

    def test_100_layers_collected(self):
        """FR1.7: Trees with up to 100 layers are fully collected."""
        # Build a wide flat tree: root + 99 children
        children = [self._make_layer(f"child_{i}") for i in range(99)]
        root = self._make_layer("root", children=children)
        result = collect_layers_depth_first(root)
        assert len(result) == 100

    def test_z_order_reflects_dfs_order(self):
        """z_order values match the DFS traversal order exactly."""
        root = self._make_layer("A", children=[
            self._make_layer("B", children=[
                self._make_layer("D"),
                self._make_layer("E"),
            ]),
            self._make_layer("C"),
        ])
        result = collect_layers_depth_first(root)
        # Expected DFS order: A(0), B(1), D(2), E(3), C(4)
        expected = [("A", 0), ("B", 1), ("D", 2), ("E", 3), ("C", 4)]
        for node, (expected_id, expected_z) in zip(result, expected):
            assert node["id"] == expected_id
            assert node["z_order"] == expected_z


# ── _truncate_layer_tree (DFS-based summary) ──────────────────────────────────

class TestTruncateLayerTree:
    """Tests for the updated _truncate_layer_tree() that uses DFS traversal."""

    @staticmethod
    def _make_layer(id: str, name: str = "", children: list | None = None,
                    **kwargs) -> dict:
        node: dict = {"id": id, "name": name or id}
        node.update(kwargs)
        if children is not None:
            node["children"] = children
        return node

    def test_returns_string(self):
        """_truncate_layer_tree always returns a string."""
        root = self._make_layer("root")
        result = _truncate_layer_tree(root)
        assert isinstance(result, str)

    def test_depth_and_z_order_in_output(self):
        """Output JSON contains depth and z_order fields."""
        import json
        root = self._make_layer("A", children=[self._make_layer("B")])
        result = _truncate_layer_tree(root)
        # Remove truncation marker if present
        clean = result.replace("\n... (truncated)", "")
        data = json.loads(clean)
        assert any("depth" in entry for entry in data)
        assert any("z_order" in entry for entry in data)

    def test_blobs_stripped(self):
        """exportedImageBase64, exportedSvg, fullPngBase64 are stripped."""
        root = self._make_layer(
            "A",
            exportedImageBase64="AAAA",
            exportedSvg="<svg/>",
            fullPngBase64="BBBB",
        )
        result = _truncate_layer_tree(root)
        assert "exportedImageBase64" not in result
        assert "exportedSvg" not in result
        assert "fullPngBase64" not in result

    def test_truncation_applied(self):
        """Output is truncated to max_chars and appends truncation marker."""
        # Build a tree large enough to exceed a tiny max_chars
        children = [self._make_layer(f"child_{i}", x=i, y=i, width=100, height=50)
                    for i in range(50)]
        root = self._make_layer("root", children=children)
        result = _truncate_layer_tree(root, max_chars=200)
        assert len(result) <= 200 + len("\n... (truncated)")
        assert result.endswith("\n... (truncated)")

    def test_no_truncation_marker_when_short(self):
        """No truncation marker when output fits within max_chars."""
        root = self._make_layer("A")
        result = _truncate_layer_tree(root, max_chars=10000)
        assert not result.endswith("\n... (truncated)")

    def test_invisible_layers_excluded(self):
        """Invisible layers are excluded from the summary."""
        root = self._make_layer("A", children=[
            {**self._make_layer("B"), "visible": False},
            self._make_layer("C"),
        ])
        result = _truncate_layer_tree(root)
        assert '"B"' not in result or '"id": "B"' not in result


# ── calculate_element_bounds ──────────────────────────────────────────────────

class TestCalculateElementBounds:
    """Tests for calculate_element_bounds()."""

    FRAME_W = 390
    FRAME_H = 844

    # ── Basic relative x/y path ───────────────────────────────────────────────

    def test_basic_relative_coords(self):
        """Relative x/y/width/height are returned as integers within bounds."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["width"] == 100
        assert result["height"] == 50

    def test_return_keys_present(self):
        """Result always contains x, y, width, height, and bounds_normalized."""
        layer = {"x": 0, "y": 0, "width": 50, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert "x" in result
        assert "y" in result
        assert "width" in result
        assert "height" in result
        assert "bounds_normalized" in result

    def test_bounds_normalized_keys(self):
        """bounds_normalized contains left, top, right, bottom."""
        layer = {"x": 0, "y": 0, "width": 50, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        assert "left" in bn
        assert "top" in bn
        assert "right" in bn
        assert "bottom" in bn

    def test_bounds_normalized_values_range(self):
        """All bounds_normalized values are in [0.0, 1.0]."""
        layer = {"x": 50, "y": 100, "width": 200, "height": 300}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        for key, val in bn.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_bounds_normalized_full_frame(self):
        """A full-frame element has normalized bounds of 0.0 and 1.0."""
        layer = {"x": 0, "y": 0, "width": self.FRAME_W, "height": self.FRAME_H}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        assert bn["left"] == pytest.approx(0.0)
        assert bn["top"] == pytest.approx(0.0)
        assert bn["right"] == pytest.approx(1.0)
        assert bn["bottom"] == pytest.approx(1.0)

    def test_bounds_normalized_center_element(self):
        """A centered element has correct normalized bounds."""
        # Element at x=195, y=422, width=100, height=100 on 390×844 frame
        layer = {"x": 195, "y": 422, "width": 100, "height": 100}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        assert bn["left"] == pytest.approx(195 / 390)
        assert bn["top"] == pytest.approx(422 / 844)
        assert bn["right"] == pytest.approx((195 + 100) / 390)
        assert bn["bottom"] == pytest.approx((422 + 100) / 844)

    def test_return_types_are_int(self):
        """x, y, width, height are Python ints."""
        layer = {"x": 10.7, "y": 20.3, "width": 100.5, "height": 50.4}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        for key in ("x", "y", "width", "height"):
            assert isinstance(result[key], int), f"{key} should be int"

    def test_bounds_normalized_types_are_float(self):
        """bounds_normalized values are floats."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        for key, val in result["bounds_normalized"].items():
            assert isinstance(val, float), f"{key} should be float"

    # ── absoluteBoundingBox path ──────────────────────────────────────────────

    def test_absolute_bounding_box_used_when_present(self):
        """absoluteBoundingBox takes precedence over relative x/y."""
        layer = {
            "x": 999,  # should be ignored
            "y": 999,
            "width": 999,
            "height": 999,
            "absoluteBoundingBox": {"x": 50, "y": 100, "width": 200, "height": 80},
        }
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 50
        assert result["y"] == 100
        assert result["width"] == 200
        assert result["height"] == 80

    def test_absolute_bounding_box_with_frame_origin(self):
        """Canvas coords are converted to frame-relative by subtracting frameOrigin."""
        # Frame starts at canvas (100, 200); element at canvas (150, 250)
        # → frame-relative: x=50, y=50
        layer = {
            "absoluteBoundingBox": {"x": 150, "y": 250, "width": 100, "height": 60},
            "frameOrigin": {"x": 100, "y": 200},
        }
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 50
        assert result["y"] == 50
        assert result["width"] == 100
        assert result["height"] == 60

    def test_absolute_bounding_box_no_frame_origin_defaults_to_zero(self):
        """When frameOrigin is absent, canvas coords are used as-is (origin = 0,0)."""
        layer = {
            "absoluteBoundingBox": {"x": 30, "y": 40, "width": 80, "height": 60},
        }
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 30
        assert result["y"] == 40

    def test_absolute_bounding_box_negative_after_origin_subtraction_clamped(self):
        """Negative frame-relative coords (after origin subtraction) are clamped to 0."""
        # Element at canvas (50, 60), frame origin at (100, 100)
        # → frame-relative: x=-50, y=-40 → clamped to 0
        layer = {
            "absoluteBoundingBox": {"x": 50, "y": 60, "width": 80, "height": 60},
            "frameOrigin": {"x": 100, "y": 100},
        }
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0
        assert result["y"] == 0

    # ── Export scale factor ───────────────────────────────────────────────────

    def test_export_scale_2x_halves_coordinates(self):
        """2× export scale divides all coordinates by 2."""
        layer = {"x": 20, "y": 40, "width": 200, "height": 100}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=2.0)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["width"] == 100
        assert result["height"] == 50

    def test_export_scale_3x(self):
        """3× export scale divides all coordinates by 3."""
        layer = {"x": 30, "y": 60, "width": 300, "height": 150}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=3.0)
        assert result["x"] == 10
        assert result["y"] == 20
        assert result["width"] == 100
        assert result["height"] == 50

    def test_export_scale_1x_no_change(self):
        """1× export scale (default) does not alter coordinates."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result_default = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        result_explicit = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=1.0)
        assert result_default["x"] == result_explicit["x"]
        assert result_default["y"] == result_explicit["y"]
        assert result_default["width"] == result_explicit["width"]
        assert result_default["height"] == result_explicit["height"]

    def test_export_scale_zero_treated_as_one(self):
        """export_scale=0 (invalid) is treated as 1.0 to avoid division by zero."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=0)
        assert result["x"] == 10
        assert result["y"] == 20

    def test_export_scale_negative_treated_as_one(self):
        """Negative export_scale is treated as 1.0."""
        layer = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=-2.0)
        assert result["x"] == 10
        assert result["y"] == 20

    def test_export_scale_with_absolute_bounding_box(self):
        """Export scale is applied to absoluteBoundingBox coordinates too."""
        layer = {
            "absoluteBoundingBox": {"x": 40, "y": 80, "width": 200, "height": 100},
        }
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H, export_scale=2.0)
        assert result["x"] == 20
        assert result["y"] == 40
        assert result["width"] == 100
        assert result["height"] == 50

    # ── Clamping and edge cases ───────────────────────────────────────────────

    def test_negative_x_clamped(self):
        """Negative x is clamped to 0."""
        layer = {"x": -10, "y": 20, "width": 100, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0

    def test_x_beyond_frame_clamped(self):
        """x beyond frame_width - 1 is clamped."""
        layer = {"x": 500, "y": 10, "width": 50, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == self.FRAME_W - 1

    def test_width_clamped_to_frame_right_edge(self):
        """x + width does not exceed frame_width."""
        layer = {"x": 300, "y": 10, "width": 200, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] + result["width"] <= self.FRAME_W

    def test_zero_width_becomes_one(self):
        """Zero width is corrected to 1."""
        layer = {"x": 10, "y": 10, "width": 0, "height": 50}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["width"] >= 1

    def test_missing_keys_default_safely(self):
        """Missing x/y/width/height default to 0 (width/height become 1)."""
        result = calculate_element_bounds({}, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] >= 1
        assert result["height"] >= 1

    def test_bounds_normalized_right_gt_left(self):
        """bounds_normalized right is always >= left."""
        layer = {"x": 50, "y": 50, "width": 100, "height": 100}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        assert bn["right"] >= bn["left"]

    def test_bounds_normalized_bottom_gt_top(self):
        """bounds_normalized bottom is always >= top."""
        layer = {"x": 50, "y": 50, "width": 100, "height": 100}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        bn = result["bounds_normalized"]
        assert bn["bottom"] >= bn["top"]

    def test_float_coords_rounded(self):
        """Float coordinates are rounded to nearest integer (±1px accuracy)."""
        layer = {"x": 10.4, "y": 20.6, "width": 100.3, "height": 50.7}
        result = calculate_element_bounds(layer, self.FRAME_W, self.FRAME_H)
        assert result["x"] == 10
        assert result["y"] == 21
        assert result["width"] == 100
        assert result["height"] == 51


# ── calculate_relative_size ───────────────────────────────────────────────────

class TestCalculateRelativeSize:
    """Tests for calculate_relative_size()."""

    FRAME_W = 390
    FRAME_H = 844

    def test_full_frame_element_returns_one(self):
        """An element covering the entire frame returns 1.0."""
        element = {"width": self.FRAME_W, "height": self.FRAME_H}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == pytest.approx(1.0)

    def test_half_frame_element(self):
        """An element covering half the frame returns ~0.5."""
        element = {"width": self.FRAME_W, "height": self.FRAME_H // 2}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == pytest.approx((self.FRAME_W * (self.FRAME_H // 2)) / (self.FRAME_W * self.FRAME_H))

    def test_small_icon_returns_small_value(self):
        """A 24×24 icon on a 390×844 frame returns a small value."""
        element = {"width": 24, "height": 24}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        expected = (24 * 24) / (self.FRAME_W * self.FRAME_H)
        assert result == pytest.approx(expected)
        assert result < 0.01  # should be very small

    def test_return_type_is_float(self):
        """Return value is always a float."""
        element = {"width": 100, "height": 100}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert isinstance(result, float)

    def test_result_in_range_zero_to_one(self):
        """Result is always in [0.0, 1.0]."""
        element = {"width": 200, "height": 400}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert 0.0 <= result <= 1.0

    def test_oversized_element_clamped_to_one(self):
        """An element larger than the frame is clamped to 1.0."""
        element = {"width": 9999, "height": 9999}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == pytest.approx(1.0)

    def test_zero_width_returns_zero(self):
        """An element with zero width returns 0.0."""
        element = {"width": 0, "height": 100}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == 0.0

    def test_zero_height_returns_zero(self):
        """An element with zero height returns 0.0."""
        element = {"width": 100, "height": 0}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == 0.0

    def test_negative_width_returns_zero(self):
        """An element with negative width returns 0.0."""
        element = {"width": -50, "height": 100}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        assert result == 0.0

    def test_missing_dimensions_returns_zero(self):
        """An element with missing width/height returns 0.0."""
        result = calculate_relative_size({}, self.FRAME_W, self.FRAME_H)
        assert result == 0.0

    def test_zero_frame_area_returns_zero(self):
        """A frame with zero area returns 0.0 (no division by zero)."""
        element = {"width": 100, "height": 100}
        result = calculate_relative_size(element, 0, 0)
        assert result == 0.0

    def test_zero_frame_width_returns_zero(self):
        """A frame with zero width returns 0.0."""
        element = {"width": 100, "height": 100}
        result = calculate_relative_size(element, 0, self.FRAME_H)
        assert result == 0.0

    def test_area_ratio_calculation(self):
        """Area ratio is element_area / frame_area."""
        element = {"width": 100, "height": 100}
        result = calculate_relative_size(element, 200, 200)
        expected = (100 * 100) / (200 * 200)  # 0.25
        assert result == pytest.approx(expected)

    def test_float_dimensions_accepted(self):
        """Float width/height values are accepted and produce correct ratio."""
        element = {"width": 195.0, "height": 422.0}
        result = calculate_relative_size(element, self.FRAME_W, self.FRAME_H)
        expected = (195.0 * 422.0) / (self.FRAME_W * self.FRAME_H)
        assert result == pytest.approx(min(1.0, expected))

    def test_square_frame_quarter_element(self):
        """A quarter-sized element on a square frame returns 0.25."""
        element = {"width": 50, "height": 50}
        result = calculate_relative_size(element, 100, 100)
        assert result == pytest.approx(0.25)


# ── detect_element_overlaps ───────────────────────────────────────────────────

class TestDetectElementOverlaps:
    """Tests for detect_element_overlaps()."""

    @staticmethod
    def _elem(layer_id: str, x: int, y: int, w: int, h: int) -> dict:
        return {"layer_id": layer_id, "x": x, "y": y, "width": w, "height": h}

    def test_no_overlap_returns_empty(self):
        """Two non-overlapping elements return []."""
        a = self._elem("a", 0, 0, 50, 50)
        b = self._elem("b", 100, 100, 50, 50)
        assert detect_element_overlaps([a, b]) == []

    def test_adjacent_elements_no_overlap(self):
        """Elements that touch at an edge (no area overlap) return []."""
        a = self._elem("a", 0, 0, 50, 50)
        b = self._elem("b", 50, 0, 50, 50)  # right edge of a == left edge of b
        assert detect_element_overlaps([a, b]) == []

    def test_partial_overlap_detected(self):
        """Two partially overlapping elements return one record with correct overlap_area."""
        a = self._elem("a", 0, 0, 100, 100)
        b = self._elem("b", 50, 50, 100, 100)
        result = detect_element_overlaps([a, b])
        assert len(result) == 1
        # Intersection: x=[50,100], y=[50,100] → 50×50 = 2500
        assert result[0]["overlap_area"] == 2500

    def test_full_containment_overlap(self):
        """Small element fully inside large element: overlap_ratio should be 1.0."""
        large = self._elem("large", 0, 0, 200, 200)
        small = self._elem("small", 50, 50, 50, 50)
        result = detect_element_overlaps([large, small])
        assert len(result) == 1
        assert result[0]["overlap_ratio"] == pytest.approx(1.0)

    def test_overlap_ratio_range(self):
        """overlap_ratio is always in [0.0, 1.0]."""
        a = self._elem("a", 0, 0, 100, 100)
        b = self._elem("b", 30, 30, 80, 80)
        result = detect_element_overlaps([a, b])
        for record in result:
            assert 0.0 <= record["overlap_ratio"] <= 1.0

    def test_symmetric_pairs(self):
        """Each pair appears exactly once (no duplicates)."""
        elements = [
            self._elem("a", 0, 0, 100, 100),
            self._elem("b", 50, 0, 100, 100),
            self._elem("c", 0, 50, 100, 100),
        ]
        result = detect_element_overlaps(elements)
        pairs = [(r["element_a_id"], r["element_b_id"]) for r in result]
        # No pair should appear twice
        assert len(pairs) == len(set(pairs))
        # Each pair should be ordered (a_id < b_id in terms of list position)
        for r in result:
            ids = [e["layer_id"] for e in elements]
            assert ids.index(r["element_a_id"]) < ids.index(r["element_b_id"])

    def test_empty_list_returns_empty(self):
        """Empty input returns []."""
        assert detect_element_overlaps([]) == []

    def test_single_element_returns_empty(self):
        """Single element returns []."""
        assert detect_element_overlaps([self._elem("a", 0, 0, 100, 100)]) == []

    def test_overlap_area_calculation(self):
        """Verify exact pixel area of intersection."""
        # a: [0,0] to [80,60]; b: [20,10] to [100,50]
        # intersection: x=[20,80], y=[10,50] → 60×40 = 2400
        a = self._elem("a", 0, 0, 80, 60)
        b = self._elem("b", 20, 10, 80, 40)
        result = detect_element_overlaps([a, b])
        assert len(result) == 1
        assert result[0]["overlap_area"] == 2400

    def test_element_ids_in_result(self):
        """Result records contain correct element_a_id and element_b_id."""
        a = self._elem("alpha", 0, 0, 100, 100)
        b = self._elem("beta", 50, 50, 100, 100)
        result = detect_element_overlaps([a, b])
        assert len(result) == 1
        assert result[0]["element_a_id"] == "alpha"
        assert result[0]["element_b_id"] == "beta"


# ── assign_z_indices ──────────────────────────────────────────────────────────

class TestAssignZIndices:
    """Tests for assign_z_indices()."""

    @staticmethod
    def _layer(z_order: int, depth: int, layer_id: str = "") -> dict:
        return {"layer_id": layer_id or f"l{z_order}_{depth}", "z_order": z_order, "depth": depth}

    def test_empty_list_returns_empty(self):
        """Empty input returns []."""
        assert assign_z_indices([]) == []

    def test_z_index_field_added(self):
        """Each layer gets a z_index field."""
        layers = [self._layer(0, 0), self._layer(1, 1)]
        result = assign_z_indices(layers)
        for layer in result:
            assert "z_index" in layer

    def test_later_z_order_higher_z_index(self):
        """Layer with higher z_order gets higher z_index (same depth)."""
        layers = [self._layer(0, 0, "a"), self._layer(1, 0, "b")]
        result = assign_z_indices(layers)
        z_by_id = {r["layer_id"]: r["z_index"] for r in result}
        assert z_by_id["b"] > z_by_id["a"]

    def test_deeper_depth_higher_z_index(self):
        """Same z_order, deeper depth → higher z_index."""
        layers = [self._layer(0, 0, "parent"), self._layer(0, 1, "child")]
        result = assign_z_indices(layers)
        z_by_id = {r["layer_id"]: r["z_index"] for r in result}
        assert z_by_id["child"] > z_by_id["parent"]

    def test_does_not_mutate_input(self):
        """Original list is not mutated."""
        layers = [self._layer(0, 0, "a"), self._layer(1, 1, "b")]
        originals = [dict(l) for l in layers]
        assign_z_indices(layers)
        for original, layer in zip(originals, layers):
            assert "z_index" not in layer
            assert layer == original

    def test_z_index_non_negative(self):
        """All z_index values are >= 0."""
        layers = [self._layer(i, i % 3) for i in range(5)]
        result = assign_z_indices(layers)
        for r in result:
            assert r["z_index"] >= 0

    def test_single_layer(self):
        """Single layer gets z_index of 0."""
        result = assign_z_indices([self._layer(0, 0, "only")])
        assert result[0]["z_index"] == 0

    def test_children_higher_than_parent(self):
        """A child (higher depth) has higher z_index than its parent (same z_order group)."""
        # Simulate a parent at z_order=2, depth=0 and child at z_order=3, depth=1
        layers = [
            self._layer(2, 0, "parent"),
            self._layer(3, 1, "child"),
        ]
        result = assign_z_indices(layers)
        z_by_id = {r["layer_id"]: r["z_index"] for r in result}
        assert z_by_id["child"] > z_by_id["parent"]


# ── analyze_visual_stack ──────────────────────────────────────────────────────

class TestAnalyzeVisualStack:
    """Tests for analyze_visual_stack()."""

    @staticmethod
    def _elem(layer_id: str, x: int, y: int, w: int, h: int) -> dict:
        return {"layer_id": layer_id, "x": x, "y": y, "width": w, "height": h}

    def test_empty_list_returns_empty(self):
        """Empty input returns []."""
        assert analyze_visual_stack([]) == []

    def test_overlaps_with_field_added(self):
        """Each element gets overlaps_with list."""
        elements = [self._elem("a", 0, 0, 50, 50), self._elem("b", 100, 100, 50, 50)]
        result = analyze_visual_stack(elements)
        for elem in result:
            assert "overlaps_with" in elem
            assert isinstance(elem["overlaps_with"], list)

    def test_overlap_count_field_added(self):
        """Each element gets overlap_count int."""
        elements = [self._elem("a", 0, 0, 50, 50), self._elem("b", 100, 100, 50, 50)]
        result = analyze_visual_stack(elements)
        for elem in result:
            assert "overlap_count" in elem
            assert isinstance(elem["overlap_count"], int)

    def test_non_overlapping_elements_zero_count(self):
        """Non-overlapping elements have overlap_count == 0."""
        elements = [self._elem("a", 0, 0, 50, 50), self._elem("b", 100, 100, 50, 50)]
        result = analyze_visual_stack(elements)
        for elem in result:
            assert elem["overlap_count"] == 0
            assert elem["overlaps_with"] == []

    def test_overlapping_elements_count(self):
        """Overlapping elements have correct overlap_count."""
        a = self._elem("a", 0, 0, 100, 100)
        b = self._elem("b", 50, 50, 100, 100)
        result = analyze_visual_stack([a, b])
        counts = {e["layer_id"]: e["overlap_count"] for e in result}
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_overlap_is_symmetric(self):
        """If A overlaps B, then B's overlaps_with contains A."""
        a = self._elem("a", 0, 0, 100, 100)
        b = self._elem("b", 50, 50, 100, 100)
        result = analyze_visual_stack([a, b])
        by_id = {e["layer_id"]: e for e in result}
        assert "b" in by_id["a"]["overlaps_with"]
        assert "a" in by_id["b"]["overlaps_with"]

    def test_does_not_mutate_input(self):
        """Original element dicts are not mutated."""
        a = self._elem("a", 0, 0, 100, 100)
        b = self._elem("b", 50, 50, 100, 100)
        originals = [dict(a), dict(b)]
        analyze_visual_stack([a, b])
        assert a == originals[0]
        assert b == originals[1]

    def test_overlaps_with_is_sorted(self):
        """overlaps_with list is sorted (deterministic)."""
        a = self._elem("a", 0, 0, 200, 200)
        b = self._elem("b", 10, 10, 50, 50)
        c = self._elem("c", 20, 20, 50, 50)
        result = analyze_visual_stack([a, b, c])
        by_id = {e["layer_id"]: e for e in result}
        overlaps_a = by_id["a"]["overlaps_with"]
        assert overlaps_a == sorted(overlaps_a)

    def test_single_element_no_overlaps(self):
        """Single element has empty overlaps_with and overlap_count == 0."""
        result = analyze_visual_stack([self._elem("only", 0, 0, 100, 100)])
        assert result[0]["overlaps_with"] == []
        assert result[0]["overlap_count"] == 0
