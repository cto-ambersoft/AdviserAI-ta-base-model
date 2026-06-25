"""
API-level tests (FastAPI TestClient). Autotrain is disabled via env so startup
does no network I/O.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _fake_gap_payload() -> dict:
    return {
        "meta": {
            "interval": "4h", "price_bars_requested": 2000, "price_bars_returned": 2,
            "dominance_bars_requested": 1200, "dominance_bars_returned": 2, "min_gap_size": 100.0,
        },
        "market_regime_current": {
            "timestamp": "2026-01-01T00:00:00+00:00", "btc_d": 50.0, "usdt_d": 5.0,
            "btc_d_trend": 0.1, "usdt_d_trend": -0.1, "market_regime": "BTC_STRENGTH",
        },
        "open_gaps": [],
        "all_gaps": [],
        "last_price_bar": {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
        },
    }


@pytest.fixture(autouse=True)
def _no_autotrain(monkeypatch):
    monkeypatch.setenv("MODEL_TECH_AUTOTRAIN_ENABLED", "0")


def test_health_is_open(monkeypatch):
    from model_tech.api.app import app

    with TestClient(app) as client:
        r = client.get("/v1/health")
    assert r.status_code == 200
    assert "global_model_available" in r.json()


def test_gap_info_is_cached(monkeypatch):
    import model_tech.api.routes as routes

    calls = {"n": 0}

    def fake() -> dict:
        calls["n"] += 1
        return _fake_gap_payload()

    monkeypatch.setattr(routes, "build_gap_info", fake)

    from model_tech.api.app import app

    with TestClient(app) as client:
        assert client.get("/v1/gap-info").status_code == 200
        assert client.get("/v1/gap-info").status_code == 200

    assert calls["n"] == 1  # second request served from the TTL cache


# --- I5: API-key authentication on expensive endpoints -----------------------

def _fake_gap(monkeypatch):
    import model_tech.api.routes as routes
    monkeypatch.setattr(routes, "build_gap_info", lambda: _fake_gap_payload())


def test_no_key_configured_leaves_endpoints_open(monkeypatch):
    monkeypatch.delenv("MODEL_TECH_API_KEY", raising=False)
    _fake_gap(monkeypatch)
    from model_tech.api.app import app

    with TestClient(app) as client:
        assert client.get("/v1/gap-info").status_code == 200


def test_protected_endpoints_require_key_when_configured(monkeypatch):
    monkeypatch.setenv("MODEL_TECH_API_KEY", "s3cret")
    _fake_gap(monkeypatch)
    from model_tech.api.app import app

    with TestClient(app) as client:
        # health stays open (monitoring/polling friendly)
        assert client.get("/v1/health").status_code == 200

        # gap-info: missing / wrong / correct key
        assert client.get("/v1/gap-info").status_code == 401
        assert client.get("/v1/gap-info", headers={"X-API-Key": "nope"}).status_code == 401
        assert client.get("/v1/gap-info", headers={"X-API-Key": "s3cret"}).status_code == 200

        # predict is rejected before any data/network work when the key is missing
        assert client.get("/v1/predict", params={"symbol": "BTCUSDT"}).status_code == 401
