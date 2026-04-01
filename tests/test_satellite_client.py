"""
Test Suite – Satellite Client (backend/satellite_client.py)
============================================================
Tests that the module uses the correct war-period date ranges and that
`fetch_orbital_data` downloads and returns the expected four file paths.
All Google Earth Engine and HTTP calls are fully mocked.

conftest.py stubs `ee` (and other unavailable packages) before collection.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend.satellite_client as sc


# ---------------------------------------------------------------------------
# Constants / date-range tests
# ---------------------------------------------------------------------------

class TestWarPeriodDates:
    """The date constants must reflect the March 2026 conflict window."""

    def test_pre_war_start_is_2025(self):
        assert sc.PRE_WAR_START == "2025-01-01"

    def test_pre_war_end_is_before_march_2026(self):
        assert sc.PRE_WAR_END == "2026-02-28"

    def test_post_war_start_is_march_2026(self):
        assert sc.POST_WAR_START == "2026-03-01"

    def test_post_war_end_is_in_march_2026(self):
        # End date should be within the conflict month
        assert sc.POST_WAR_END.startswith("2026-03")

    def test_pre_war_ends_before_post_war_starts(self):
        """Baseline must not overlap with the post-war window."""
        assert sc.PRE_WAR_END < sc.POST_WAR_START


# ---------------------------------------------------------------------------
# fetch_orbital_data – mocked GEE + HTTP
# ---------------------------------------------------------------------------

def _make_ee_mock():
    """Build a minimal Earth Engine mock that avoids real API calls."""
    ee_mock = MagicMock()
    # Geometry chain
    point_mock  = MagicMock()
    buffer_mock = MagicMock()
    bounds_mock = MagicMock()
    ee_mock.Geometry.Point.return_value  = point_mock
    point_mock.buffer.return_value       = buffer_mock
    buffer_mock.bounds.return_value      = bounds_mock

    # ImageCollection chain
    ic_mock = MagicMock()
    ee_mock.ImageCollection.return_value = ic_mock
    ic_mock.filterBounds.return_value    = ic_mock
    ic_mock.filterDate.return_value      = ic_mock

    filter_mock = MagicMock()
    ee_mock.Filter.lt.return_value = filter_mock
    ic_mock.filter.return_value = ic_mock

    image_mock = MagicMock()
    ic_mock.median.return_value          = image_mock
    image_mock.visualize.return_value    = image_mock
    image_mock.clip.return_value         = image_mock
    image_mock.normalizedDifference.return_value = image_mock
    image_mock.rename.return_value       = image_mock
    image_mock.subtract.return_value     = image_mock
    image_mock.getThumbURL.return_value  = "https://earthengine.googleapis.com/thumb/test"
    image_mock.select.return_value       = image_mock

    return ee_mock


class TestFetchOrbitalData:
    @pytest.fixture()
    def tmp_output(self, tmp_path):
        return str(tmp_path)

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_returns_four_paths(self, mock_ee, mock_get, tmp_output):
        mock_ee.__dict__.update(_make_ee_mock().__dict__)
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        result = sc.fetch_orbital_data(27.18, 56.27, 1, tmp_output)
        assert len(result) == 4

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_pre_optical_file_created(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        pre, post, ndvi_delta, ndvi_pre = sc.fetch_orbital_data(27.18, 56.27, 42, tmp_output)
        assert os.path.exists(pre)
        assert "pre_optical_42" in pre

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_post_optical_file_created(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        pre, post, ndvi_delta, ndvi_pre = sc.fetch_orbital_data(27.18, 56.27, 42, tmp_output)
        assert os.path.exists(post)
        assert "post_optical_42" in post

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_ndvi_delta_file_created(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        pre, post, ndvi_delta, ndvi_pre = sc.fetch_orbital_data(27.18, 56.27, 42, tmp_output)
        assert os.path.exists(ndvi_delta)
        assert "real_ndvi_42" in ndvi_delta

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_ndvi_pre_file_created(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        pre, post, ndvi_delta, ndvi_pre = sc.fetch_orbital_data(27.18, 56.27, 42, tmp_output)
        assert os.path.exists(ndvi_pre)
        assert "ndvi_pre_42" in ndvi_pre

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_four_http_requests_made(self, mock_ee, mock_get, tmp_output):
        """One request per file: pre RGB, post RGB, pre NDVI, NDVI delta."""
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        sc.fetch_orbital_data(27.18, 56.27, 99, tmp_output)
        assert mock_get.call_count == 4

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_roi_centred_on_supplied_coordinates(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        sc.fetch_orbital_data(32.08, 34.78, 5, tmp_output)
        mock_ee.Geometry.Point.assert_called_once_with([34.78, 32.08])

    @patch("backend.satellite_client.requests.get")
    @patch("backend.satellite_client.ee")
    def test_2500m_buffer_applied(self, mock_ee, mock_get, tmp_output):
        _setup_ee_mock(mock_ee)
        mock_get.return_value.content = b"\x89PNG fake"

        sc.fetch_orbital_data(32.08, 34.78, 5, tmp_output)
        # The buffer call should use 2500 (meters)
        point_mock = mock_ee.Geometry.Point.return_value
        point_mock.buffer.assert_called_once_with(2500)


# ---------------------------------------------------------------------------
# initialize_engine
# ---------------------------------------------------------------------------

class TestInitializeEngine:
    @patch("backend.satellite_client.ee")
    def test_initializes_with_project_id(self, mock_ee):
        sc.initialize_engine()
        mock_ee.Initialize.assert_called_once_with(project="orbital-intelligence-thesis")

    @patch("backend.satellite_client.ee")
    def test_raises_on_auth_failure(self, mock_ee):
        mock_ee.Initialize.side_effect = Exception("auth failed")
        with pytest.raises(Exception, match="auth failed"):
            sc.initialize_engine()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _setup_ee_mock(mock_ee):
    """Populate mock_ee with the full Earth Engine call chain."""
    ee_real = _make_ee_mock()
    for attr, value in vars(ee_real).items():
        setattr(mock_ee, attr, value)

    # Geometry
    point_mock  = MagicMock()
    buffer_mock = MagicMock()
    bounds_mock = MagicMock()
    mock_ee.Geometry.Point.return_value  = point_mock
    point_mock.buffer.return_value       = buffer_mock
    buffer_mock.bounds.return_value      = bounds_mock

    # ImageCollection chain
    ic_mock = MagicMock()
    mock_ee.ImageCollection.return_value = ic_mock
    ic_mock.filterBounds.return_value    = ic_mock
    ic_mock.filterDate.return_value      = ic_mock

    filter_mock = MagicMock()
    mock_ee.Filter.lt.return_value = filter_mock
    ic_mock.filter.return_value = ic_mock

    image_mock = MagicMock()
    ic_mock.median.return_value                   = image_mock
    image_mock.visualize.return_value             = image_mock
    image_mock.clip.return_value                  = image_mock
    image_mock.normalizedDifference.return_value  = image_mock
    image_mock.rename.return_value                = image_mock
    image_mock.subtract.return_value              = image_mock
    image_mock.select.return_value                = image_mock
    image_mock.getThumbURL.return_value           = "https://earthengine.googleapis.com/thumb/test"
