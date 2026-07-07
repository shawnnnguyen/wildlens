"""
Unit tests for fetcher.py's caching behavior (bug #4): a transient network
error must never be cached as if it were a confirmed empty/not-found result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildlens.data import fetcher


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    dirs = {
        "_CACHE_EOL":        tmp_path / "eol",
        "_CACHE_IUCN":       tmp_path / "iucn",
        "_CACHE_WIKIPEDIA":  tmp_path / "wikipedia",
        "_CACHE_API_NINJAS": tmp_path / "api_ninjas",
    }
    for name, path in dirs.items():
        monkeypatch.setattr(fetcher, name, path)
    return dirs


def _connect_error(*_args, **_kwargs):
    raise httpx.ConnectError("boom")


# ── EOLClient ─────────────────────────────────────────────────────────────────

def test_eol_transient_error_not_cached(cache_dirs):
    client = fetcher.EOLClient()
    with patch.object(client._client, "get", side_effect=_connect_error):
        result = client.fetch("African Lion", "Panthera leo")
    assert result == {}
    assert not list(cache_dirs["_CACHE_EOL"].glob("*.json"))


def test_eol_confirmed_not_found_is_cached(cache_dirs):
    client = fetcher.EOLClient()
    empty_resp = MagicMock()
    empty_resp.raise_for_status.return_value = None
    empty_resp.json.return_value = {"results": []}
    with patch.object(client._client, "get", return_value=empty_resp):
        result = client.fetch("Nonexistent", "Nonexistentus fakeus")
    assert result == {}
    assert list(cache_dirs["_CACHE_EOL"].glob("*.json"))


# ── APINinjasClient ───────────────────────────────────────────────────────────

def test_api_ninjas_transient_error_not_cached(cache_dirs, monkeypatch):
    monkeypatch.setenv("API_NINJAS_KEY", "test-key")
    client = fetcher.APINinjasClient()
    with patch.object(client._client, "get", side_effect=_connect_error):
        result = client.fetch("African Lion")
    assert result == {}
    assert not list(cache_dirs["_CACHE_API_NINJAS"].glob("*.json"))


def test_api_ninjas_confirmed_not_found_is_cached(cache_dirs, monkeypatch):
    monkeypatch.setenv("API_NINJAS_KEY", "test-key")
    client = fetcher.APINinjasClient()
    empty_resp = MagicMock()
    empty_resp.raise_for_status.return_value = None
    empty_resp.json.return_value = []
    with patch.object(client._client, "get", return_value=empty_resp):
        result = client.fetch("Nonexistent")
    assert result == {}
    assert list(cache_dirs["_CACHE_API_NINJAS"].glob("*.json"))


# ── IUCNClient ────────────────────────────────────────────────────────────────

def test_iucn_transient_error_not_cached_and_all_legs_attempted(cache_dirs, monkeypatch):
    monkeypatch.setenv("IUCN_API_KEY", "test-key")
    client = fetcher.IUCNClient()
    with patch.object(client._client, "get", side_effect=_connect_error) as mock_get:
        result = client.fetch("Panthera leo")
    assert mock_get.call_count == 3  # assessment, threats, habitats all attempted
    assert result["category"] == "NE"
    assert not list(cache_dirs["_CACHE_IUCN"].glob("*.json"))


def test_iucn_assessment_failure_does_not_abort_other_legs(cache_dirs, monkeypatch):
    monkeypatch.setenv("IUCN_API_KEY", "test-key")
    client = fetcher.IUCNClient()

    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None
    ok_resp.json.return_value = {"result": [{"title": "Habitat loss"}, {"habitat": "Savanna"}]}

    calls = {"n": 0}

    def _get(url, *_args, **_kwargs):
        calls["n"] += 1
        if "species/" in url and "threats" not in url and "habitats" not in url:
            raise httpx.ConnectError("boom")
        return ok_resp

    with patch.object(client._client, "get", side_effect=_get):
        result, transient = client._fetch_live("Panthera leo")

    assert calls["n"] == 3
    assert transient is True


def test_iucn_success_is_cached(cache_dirs, monkeypatch):
    monkeypatch.setenv("IUCN_API_KEY", "test-key")
    client = fetcher.IUCNClient()
    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None
    ok_resp.json.return_value = {"result": []}
    with patch.object(client._client, "get", return_value=ok_resp):
        client.fetch("Panthera leo")
    assert list(cache_dirs["_CACHE_IUCN"].glob("*.json"))


# ── WikipediaClient ───────────────────────────────────────────────────────────

def test_wikipedia_transient_error_not_cached(cache_dirs):
    client = fetcher.WikipediaClient()
    with patch.object(client._client, "get", side_effect=_connect_error):
        result = client.fetch("African Lion", "Panthera leo")
    assert result == {}
    assert not list(cache_dirs["_CACHE_WIKIPEDIA"].glob("*.json"))


def test_wikipedia_confirmed_not_found_is_cached(cache_dirs):
    client = fetcher.WikipediaClient()
    missing_resp = MagicMock()
    missing_resp.raise_for_status.return_value = None
    missing_resp.json.return_value = {"query": {"pages": {"-1": {"missing": ""}}}}
    with patch.object(client._client, "get", return_value=missing_resp):
        result = client.fetch("Nonexistent", "Nonexistentus fakeus")
    assert result == {}
    assert list(cache_dirs["_CACHE_WIKIPEDIA"].glob("*.json"))
