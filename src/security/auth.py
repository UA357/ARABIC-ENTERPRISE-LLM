import os
import secrets

from fastapi import HTTPException, Request


API_KEY = os.getenv("ARABIC_LLM_API_KEY", "")
ADMIN_API_KEY = os.getenv("ARABIC_LLM_ADMIN_API_KEY", "")


def authenticate_request(request: Request) -> str:
    """
    Authenticate a request using a Bearer API key.

    Returns:
        The authenticated role: "generator" or "admin".
    """

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Authentication is not configured.",
        )

    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Admin authentication is not configured.",
        )

    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if secrets.compare_digest(token, ADMIN_API_KEY):
        return "admin"

    if secrets.compare_digest(token, API_KEY):
        return "generator"

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def authorize_generate(request: Request) -> str:
    """
    Authorization boundary for protected generation endpoints.

    Both generator and admin roles may generate.
    """
    role = authenticate_request(request)

    if role not in {"generator", "admin"}:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions.",
        )

    return role


def authorize_admin(request: Request) -> str:
    """
    Authorization boundary for admin-only endpoints.
    """
    role = authenticate_request(request)

    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required.",
        )

    return role
