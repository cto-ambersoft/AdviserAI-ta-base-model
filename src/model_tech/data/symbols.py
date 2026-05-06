from __future__ import annotations


_KNOWN_QUOTES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
    "TRY",
    "GBP",
)


def normalize_symbol(symbol: str, *, default_quote: str = "USDT") -> str:
    """
    Normalize user input into a Binance Spot symbol (e.g. BTCUSDT).

    Supported inputs:
    - "BTC" -> "BTCUSDT" (default_quote appended)
    - "BTCUSDT" -> "BTCUSDT"
    - "btc/usdt" -> "BTCUSDT"
    - "btc-usdt" -> "BTCUSDT"
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s

    # Common separators
    for sep in ("/", "-", "_", " "):
        if sep in s:
            parts = [p for p in s.split(sep) if p]
            if len(parts) == 2:
                base, quote = parts[0], parts[1]
                return f"{base}{quote}".upper()
            # fall through to "remove separators" behavior
            s = s.replace(sep, "")

    dq = (default_quote or "USDT").strip().upper()
    if not dq:
        dq = "USDT"

    # If it's already a pair (ends with a known quote and has a base part), keep.
    for q in _KNOWN_QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s

    # Otherwise interpret as base-asset and append default quote.
    return f"{s}{dq}"


