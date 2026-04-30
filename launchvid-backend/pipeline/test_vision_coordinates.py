"""
Unit tests for extract_precise_coordinates() and validate_element_coordinates()
in pipeline/vision.py.

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
            from pipeline.vision import extract_precise_coordinates, validate_element_coordinates


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
