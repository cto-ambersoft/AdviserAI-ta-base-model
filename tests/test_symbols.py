"""I6 — coverage for symbol normalization (used on every API/CLI entry)."""
from __future__ import annotations

import pytest

from model_tech.data.symbols import normalize_symbol, validate_symbol


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC", "BTCUSDT"),
        ("btc/usdt", "BTCUSDT"),
        ("btc-usdt", "BTCUSDT"),
        ("BTC_USDT", "BTCUSDT"),
        ("BTCUSDT", "BTCUSDT"),
        ("  eth  ", "ETHUSDT"),
        ("", ""),
    ],
)
def test_normalize_symbol(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


def test_default_quote_is_applied() -> None:
    assert normalize_symbol("eth", default_quote="BTC") == "ETHBTC"
    assert normalize_symbol("sol", default_quote="") == "SOLUSDT"  # falls back to USDT


def test_path_separators_are_stripped() -> None:
    # Guards the filesystem path built from the symbol (data/<SYMBOL>_4h.parquet).
    for raw in ["../../etc", "a/b/c", "x/y"]:
        assert "/" not in normalize_symbol(raw)


# --- S8: explicit ^[A-Z0-9]+$ validation (defense-in-depth) -------------------

def test_validate_symbol_accepts_canonical() -> None:
    assert validate_symbol("BTCUSDT") == "BTCUSDT"
    assert validate_symbol("1000SHIBUSDT") == "1000SHIBUSDT"
    assert validate_symbol("btcusdt") == "BTCUSDT"  # normalizes case


@pytest.mark.parametrize(
    "bad",
    ["", "BTC/USDT", "../etc", "BTC USDT", "BTC.D", "X\\Y", "..", "BTC-USDT", "A" * 40],
)
def test_validate_symbol_rejects_unsafe(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_symbol(bad)


def test_symbol_path_helpers_reject_traversal(tmp_path) -> None:
    from model_tech.artifacts import artifact_paths
    from model_tech.config import Paths
    from model_tech.data.store import symbol_ohlcv_path

    paths = Paths(root=tmp_path, data_dir_override=tmp_path, artifacts_dir_override=tmp_path)
    with pytest.raises(ValueError):
        symbol_ohlcv_path(paths, "../evil")
    with pytest.raises(ValueError):
        artifact_paths(paths, model_id="../evil")
