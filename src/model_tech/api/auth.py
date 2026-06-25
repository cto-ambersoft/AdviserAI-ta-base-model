from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"

# auto_error=False: a missing header yields None instead of erroring, so we can
# treat "no key configured" as open access and compare in constant time ourselves.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(request: Request, provided: str | None = Depends(_api_key_header)) -> None:
    """
    Guard expensive/mutating endpoints with a shared API key.

    The expected key is loaded from MODEL_TECH_API_KEY into app.state at startup.
    If it is empty (unset), authentication is disabled (local/dev default). When
    set, the request must carry a matching `X-API-Key` header.
    """
    expected = (getattr(request.app.state, "api_key", "") or "")
    if not expected:
        return  # auth disabled
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
