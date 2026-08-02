"""Supabase JWT auth for the API (WIN 7c.2).

The frontend attaches the user's Supabase access token as `Authorization: Bearer <jwt>`.
We validate it by asking Supabase's auth server (`auth.get_user(jwt)`) instead of decoding the
JWT ourselves — no signing-secret to manage, and token revocation is honoured. The verified
`user.id` is the ONLY source of identity: request bodies never carry a user_id, so a client
cannot impersonate another user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from src.storage.supabase_store import get_store

logger = logging.getLogger("rag.api.auth")


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None


def get_current_user(authorization: str | None = Header(default=None)) -> AuthUser:
    """FastAPI dependency: resolve the caller from a validated Supabase bearer token.

    Raises 401 if the token is absent/invalid, 503 if Supabase isn't configured (auth can't
    work without it). Never leaks validation details to the client.
    """
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="auth unavailable"
        )

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token"
        )
    token = authorization.split(" ", 1)[1].strip()

    try:
        res = store.client.auth.get_user(token)
    except Exception:  # noqa: BLE001 - any auth/client/network failure means the token is unusable
        logger.info("token validation failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    user = getattr(res, "user", None)
    if user is None or not getattr(user, "id", None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return AuthUser(id=user.id, email=getattr(user, "email", None))
