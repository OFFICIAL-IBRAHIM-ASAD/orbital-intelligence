"""
Test Suite – Backend API (main.py)
===================================
Tests all three FastAPI endpoints using the TestClient.

External dependencies (Earth Engine, PyTorch model) are mocked via @patch
decorators rather than sys.modules replacement, so they do not pollute the
module cache for other test files in the same pytest session.

conftest.py (in this directory) pre-loads stubs for ee, torch, cv2, etc.
before any test module is collected.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import backend.main as app_module

client = TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Helper – reset ledger to a known state before each test
# ---------------------------------------------------------------------------

ORIGINAL_LEDGER = [
    {
        "id": 1,
        "claim_type": "Structural Survey",
        "description": "Test site near Bandar Abbas, Iran.",
        "status": "Pending Verification",
        "latitude": 27.1832,
        "longitude": 56.2666,
    }
]


@pytest.fixture(autouse=True)
def reset_ledger():
    """Restore global_ledger to a clean state before every test."""
    app_module.global_ledger.clear()
    app_module.global_ledger.extend(
        [{**entry} for entry in ORIGINAL_LEDGER]
    )
    yield
    app_module.global_ledger.clear()
    app_module.global_ledger.extend(
        [{**entry} for entry in ORIGINAL_LEDGER]
    )


# ---------------------------------------------------------------------------
# GET /api/claims
# ---------------------------------------------------------------------------

class TestGetAllClaims:
    def test_returns_200(self):
        response = client.get("/api/claims")
        assert response.status_code == 200

    def test_returns_list(self):
        response = client.get("/api/claims")
        assert isinstance(response.json(), list)

    def test_initial_ledger_has_one_claim(self):
        response = client.get("/api/claims")
        data = response.json()
        assert len(data) == 1

    def test_claim_has_required_fields(self):
        data = client.get("/api/claims").json()
        claim = data[0]
        assert "id"          in claim
        assert "claim_type"  in claim
        assert "description" in claim
        assert "status"      in claim
        assert "latitude"    in claim
        assert "longitude"   in claim

    def test_initial_claim_status_is_pending(self):
        data = client.get("/api/claims").json()
        assert data[0]["status"] == "Pending Verification"


# ---------------------------------------------------------------------------
# POST /api/target
# ---------------------------------------------------------------------------

class TestCreateNewTarget:
    @patch("backend.main.get_place_name", return_value="Haifa, Israel")
    def test_creates_new_claim(self, mock_geocode):
        payload = {"latitude": 32.8191, "longitude": 34.9983}
        response = client.post("/api/target", json=payload)
        assert response.status_code == 200

    @patch("backend.main.get_place_name", return_value="Haifa, Israel")
    def test_response_contains_coordinates(self, mock_geocode):
        payload = {"latitude": 32.8191, "longitude": 34.9983}
        data = client.post("/api/target", json=payload).json()
        assert data["latitude"]  == pytest.approx(32.8191)
        assert data["longitude"] == pytest.approx(34.9983)

    @patch("backend.main.get_place_name", return_value="Haifa, Israel")
    def test_new_claim_appended_to_ledger(self, mock_geocode):
        payload = {"latitude": 32.8191, "longitude": 34.9983}
        client.post("/api/target", json=payload)
        claims = client.get("/api/claims").json()
        assert len(claims) == 2

    @patch("backend.main.get_place_name", return_value="Haifa, Israel")
    def test_new_claim_has_unique_id(self, mock_geocode):
        payload = {"latitude": 32.8191, "longitude": 34.9983}
        data = client.post("/api/target", json=payload).json()
        assert data["id"] > ORIGINAL_LEDGER[0]["id"]

    @patch("backend.main.get_place_name", return_value="Unknown Region")
    def test_new_claim_default_status_is_pending(self, mock_geocode):
        payload = {"latitude": 0.0, "longitude": 0.0}
        data = client.post("/api/target", json=payload).json()
        assert data["status"] == "Pending Verification"

    @patch("backend.main.get_place_name", return_value="Tehran, Iran")
    def test_place_name_appears_in_description(self, mock_geocode):
        payload = {"latitude": 35.69, "longitude": 51.39}
        data = client.post("/api/target", json=payload).json()
        assert "Tehran, Iran" in data["description"]

    def test_missing_latitude_returns_422(self):
        response = client.post("/api/target", json={"longitude": 34.9})
        assert response.status_code == 422

    def test_missing_longitude_returns_422(self):
        response = client.post("/api/target", json={"latitude": 32.8})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/analyze/{claim_id}
# ---------------------------------------------------------------------------

class TestAnalyzeDynamicTarget:
    def _mock_pipeline(self, damage_pct=12.5):
        mock_fetch = patch(
            "backend.main.fetch_orbital_data",
            return_value=(
                "/tmp/pre_optical_1.png",
                "/tmp/post_optical_1.png",
                "/tmp/real_ndvi_1.png",
                "/tmp/ndvi_pre_1.png",
            ),
        )
        mock_run = patch(
            "backend.main.run_analysis",
            return_value={"damage_percentage": damage_pct},
        )
        return mock_fetch, mock_run

    def test_unknown_claim_returns_error(self):
        response = client.get("/api/analyze/9999")
        data = response.json()
        assert "error" in data

    def test_verified_verdict_when_damage_above_threshold(self):
        mf, mr = self._mock_pipeline(damage_pct=20.0)
        with mf, mr:
            response = client.get("/api/analyze/1")
        data = response.json()
        assert data["verdict"] == "Verified"

    def test_unverified_verdict_when_damage_below_threshold(self):
        mf, mr = self._mock_pipeline(damage_pct=2.0)
        with mf, mr:
            response = client.get("/api/analyze/1")
        data = response.json()
        assert data["verdict"] == "Unverified"

    def test_response_contains_all_image_urls(self):
        mf, mr = self._mock_pipeline(damage_pct=10.0)
        with mf, mr:
            data = client.get("/api/analyze/1").json()
        assert "xai_heatmap_url"  in data
        assert "ndvi_map_url"     in data
        assert "ndvi_pre_url"     in data
        assert "pre_optical_url"  in data
        assert "post_optical_url" in data

    def test_response_contains_cost_estimate(self):
        mf, mr = self._mock_pipeline(damage_pct=10.0)
        with mf, mr:
            data = client.get("/api/analyze/1").json()
        assert "cost_estimate" in data
        ce = data["cost_estimate"]
        assert "cost_display"    in ce
        assert "region"          in ce
        assert "cost_usd"        in ce
        assert "damaged_area_m2" in ce

    def test_image_urls_reference_localhost(self):
        mf, mr = self._mock_pipeline(damage_pct=10.0)
        with mf, mr:
            data = client.get("/api/analyze/1").json()
        assert data["xai_heatmap_url"].startswith("http://localhost:8000")

    def test_details_contains_coordinates(self):
        mf, mr = self._mock_pipeline(damage_pct=10.0)
        with mf, mr:
            data = client.get("/api/analyze/1").json()
        assert "27.18" in data["details"] or "27.2" in data["details"]

    def test_pipeline_failure_returns_error(self):
        with patch("backend.main.fetch_orbital_data", side_effect=RuntimeError("EE timeout")):
            response = client.get("/api/analyze/1")
        data = response.json()
        assert "error" in data


# ---------------------------------------------------------------------------
# GET /api/cost_estimate/{claim_id}
# ---------------------------------------------------------------------------

class TestCostEstimateEndpoint:
    def _run_analysis(self, damage_pct=15.0):
        mock_fetch = patch(
            "backend.main.fetch_orbital_data",
            return_value=(
                "/tmp/pre_optical_1.png",
                "/tmp/post_optical_1.png",
                "/tmp/real_ndvi_1.png",
                "/tmp/ndvi_pre_1.png",
            ),
        )
        mock_run = patch(
            "backend.main.run_analysis",
            return_value={"damage_percentage": damage_pct},
        )
        with mock_fetch, mock_run:
            client.get("/api/analyze/1")

    def test_404_for_unknown_claim(self):
        response = client.get("/api/cost_estimate/9999")
        assert response.status_code == 404

    def test_400_before_analysis_is_run(self):
        response = client.get("/api/cost_estimate/1")
        assert response.status_code == 400

    def test_returns_cost_info_after_analysis(self):
        self._run_analysis(damage_pct=15.0)
        response = client.get("/api/cost_estimate/1")
        assert response.status_code == 200
        data = response.json()
        assert "cost_usd"          in data
        assert "cost_display"      in data
        assert "region"            in data
        assert "damage_percentage" in data

    def test_damage_percentage_matches_analysis(self):
        self._run_analysis(damage_pct=15.0)
        data = client.get("/api/cost_estimate/1").json()
        assert data["damage_percentage"] == pytest.approx(15.0)

    def test_region_detected_correctly_for_iran_claim(self):
        self._run_analysis()
        data = client.get("/api/cost_estimate/1").json()
        assert data["region"] == "Iran"


# ---------------------------------------------------------------------------
# get_place_name helper
# ---------------------------------------------------------------------------

class TestGetPlaceName:
    def test_returns_string_on_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "Bandar Abbas"}
        with patch("backend.main.requests.get", return_value=mock_response):
            result = app_module.get_place_name(27.18, 56.27)
        assert result == "Bandar Abbas"

    def test_falls_back_to_country_when_no_name(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"address": {"country": "Iran"}}
        with patch("backend.main.requests.get", return_value=mock_response):
            result = app_module.get_place_name(27.18, 56.27)
        assert result == "Iran"

    def test_returns_unknown_region_on_exception(self):
        with patch("backend.main.requests.get", side_effect=Exception("timeout")):
            result = app_module.get_place_name(0.0, 0.0)
        assert result == "Unknown Coordinates"

