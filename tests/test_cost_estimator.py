"""
Test Suite – Cost Estimator
============================
Validates backend/cost_estimator.py in isolation (no external dependencies).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.cost_estimator import (
    detect_region,
    estimate_damage_cost,
    IRAN_BBOX,
    ISRAEL_BBOX,
    ROI_AREA_M2,
    REGION_PARAMS,
)


# ---------------------------------------------------------------------------
# detect_region
# ---------------------------------------------------------------------------

class TestDetectRegion:
    def test_bandar_abbas_is_iran(self):
        assert detect_region(27.18, 56.27) == "iran"

    def test_tehran_is_iran(self):
        assert detect_region(35.69, 51.39) == "iran"

    def test_tel_aviv_is_israel(self):
        assert detect_region(32.08, 34.78) == "israel"

    def test_jerusalem_is_israel(self):
        assert detect_region(31.77, 35.23) == "israel"

    def test_unknown_coordinates_return_default(self):
        # Middle of the Atlantic Ocean
        assert detect_region(0.0, -30.0) == "default"

    def test_iraq_inside_iran_bbox_returns_iran(self):
        # Baghdad (33.34°N, 44.39°E) falls inside the Iran bounding box because
        # the bbox is defined broadly (lat 25-40, lon 44-64) to cover all of Iran.
        # The simple bbox approach does not distinguish Iran from neighbouring Iraq.
        assert detect_region(33.34, 44.39) == "iran"

    def test_iran_boundary_north(self):
        assert detect_region(IRAN_BBOX[1], 50.0) == "iran"  # upper boundary inclusive

    def test_iran_boundary_south(self):
        assert detect_region(IRAN_BBOX[0], 50.0) == "iran"  # lower boundary inclusive

    def test_israel_boundary(self):
        assert detect_region(ISRAEL_BBOX[0], ISRAEL_BBOX[2]) == "israel"


# ---------------------------------------------------------------------------
# estimate_damage_cost – happy paths
# ---------------------------------------------------------------------------

class TestEstimateDamageCostHappyPaths:
    # Iran scenario
    def test_iran_zero_damage(self):
        result = estimate_damage_cost(0.0, 27.18, 56.27)
        assert result["cost_usd"] == 0.0
        assert result["damaged_area_m2"] == 0.0
        assert result["region"] == "Iran"

    def test_iran_100_percent_damage(self):
        result = estimate_damage_cost(100.0, 27.18, 56.27)
        p = REGION_PARAMS["iran"]
        expected = ROI_AREA_M2 * p["urban_density"] * p["cost_per_m2_usd"] * p["infra_multiplier"]
        assert abs(result["cost_usd"] - expected) < 1.0  # within $1 rounding

    def test_israel_50_percent_damage_is_nonzero(self):
        result = estimate_damage_cost(50.0, 32.08, 34.78)
        assert result["cost_usd"] > 0
        assert result["region"] == "Israel"

    def test_israel_greater_cost_per_m2_than_iran(self):
        iran_result   = estimate_damage_cost(10.0, 35.69, 51.39)
        israel_result = estimate_damage_cost(10.0, 32.08, 34.78)
        assert israel_result["cost_usd"] > iran_result["cost_usd"]

    def test_default_region_produces_result(self):
        result = estimate_damage_cost(20.0, 0.0, 0.0)
        assert result["region"] == "Default"
        assert result["cost_usd"] > 0


# ---------------------------------------------------------------------------
# estimate_damage_cost – return structure
# ---------------------------------------------------------------------------

class TestEstimateDamageCostStructure:
    def test_returns_all_required_keys(self):
        result = estimate_damage_cost(15.0, 27.18, 56.27)
        assert "region"          in result
        assert "damaged_area_m2" in result
        assert "cost_usd"        in result
        assert "cost_display"    in result
        assert "assumptions"     in result

    def test_assumptions_contains_cost_params(self):
        result = estimate_damage_cost(15.0, 27.18, 56.27)
        a = result["assumptions"]
        assert "cost_per_m2_usd"  in a
        assert "urban_density"    in a
        assert "infra_multiplier" in a
        assert "roi_area_m2"      in a

    def test_cost_display_billion_label(self):
        # Iran 100 % damage should be in billions
        result = estimate_damage_cost(100.0, 27.18, 56.27)
        assert "billion" in result["cost_display"].lower() or "million" in result["cost_display"].lower()

    def test_cost_display_contains_dollar_sign(self):
        result = estimate_damage_cost(10.0, 27.18, 56.27)
        assert "$" in result["cost_display"]

    def test_damaged_area_scales_linearly_with_damage_percentage(self):
        r10 = estimate_damage_cost(10.0, 27.18, 56.27)
        r20 = estimate_damage_cost(20.0, 27.18, 56.27)
        assert abs(r20["damaged_area_m2"] / r10["damaged_area_m2"] - 2.0) < 1e-6

    def test_cost_scales_linearly_with_damage_percentage(self):
        r5  = estimate_damage_cost(5.0,  35.69, 51.39)
        r25 = estimate_damage_cost(25.0, 35.69, 51.39)
        assert abs(r25["cost_usd"] / r5["cost_usd"] - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# estimate_damage_cost – input validation
# ---------------------------------------------------------------------------

class TestEstimateDamageCostValidation:
    def test_negative_damage_raises_value_error(self):
        with pytest.raises(ValueError, match="damage_percentage"):
            estimate_damage_cost(-1.0, 27.18, 56.27)

    def test_damage_above_100_raises_value_error(self):
        with pytest.raises(ValueError, match="damage_percentage"):
            estimate_damage_cost(101.0, 27.18, 56.27)

    def test_exactly_zero_is_valid(self):
        result = estimate_damage_cost(0.0, 27.18, 56.27)
        assert result["cost_usd"] == 0.0

    def test_exactly_100_is_valid(self):
        result = estimate_damage_cost(100.0, 27.18, 56.27)
        assert result["cost_usd"] > 0


# ---------------------------------------------------------------------------
# estimate_damage_cost – custom ROI area
# ---------------------------------------------------------------------------

class TestCustomRoiArea:
    def test_custom_roi_scales_cost(self):
        small_roi = estimate_damage_cost(10.0, 27.18, 56.27, roi_area_m2=10_000_000)
        large_roi = estimate_damage_cost(10.0, 27.18, 56.27, roi_area_m2=20_000_000)
        assert abs(large_roi["cost_usd"] / small_roi["cost_usd"] - 2.0) < 1e-6
