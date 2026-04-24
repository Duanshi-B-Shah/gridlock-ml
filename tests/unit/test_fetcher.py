"""Tests for the OpenF1 data fetcher (mocked API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from f1_predictor.data.fetcher import _get, fetch_sessions


class TestApiClient:
    @patch("f1_predictor.data.fetcher.requests.get")
    def test_get_returns_json(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"key": "value"}]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get("sessions", {"year": 2024})
        assert result == [{"key": "value"}]

    @patch("f1_predictor.data.fetcher.requests.get")
    def test_get_retries_on_failure(self, mock_get):
        """Should retry on RequestException and succeed on third attempt."""
        mock_get.side_effect = [
            requests.RequestException("Connection error"),
            requests.RequestException("Connection error"),
            MagicMock(json=lambda: [{"ok": True}], raise_for_status=MagicMock()),
        ]

        result = _get("sessions")
        assert result == [{"ok": True}]
        assert mock_get.call_count == 3

    @patch("f1_predictor.data.fetcher._get")
    def test_fetch_sessions_returns_dataframe(self, mock_api):
        mock_api.return_value = [
            {"session_key": 1, "session_name": "Race", "circuit_short_name": "monza"},
            {"session_key": 2, "session_name": "Race", "circuit_short_name": "spa"},
        ]

        result = fetch_sessions(2024)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
