"""Authentication for SOPProof: Argon2id passwords, opaque server-side
sessions in HttpOnly cookies, double-submit CSRF, and rate limiting.

Only the SHA-256 of a session/invitation token is ever stored. Raw tokens
exist only in the browser cookie and, for invitations, in the single
creation response shown to the administrator.
"""
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request

from backend import store

SESSION_COOKIE = "sopproof_session"
CSRF_COOKIE = "sopproof_csrf"
CSRF_HEADER = "x-csrf-token"

_hasher = PasswordHasher()  # Argon2id, OWASP-recommended defaults
# Verified against when an account does not exist, so login timing does not
# reveal whether an email is registered.
_DUMMY_HASH = _hasher.hash("sopproof-dummy-password-for-timing")


def session_days() -> int:
    try:
        return max(1, min(30, int(os.environ.get("SOPPROOF_SESSION_DAYS", "7"))))
    except ValueError:
        return 7


def cookie_secure() -> bool:
    """Secure cookies for HTTPS deployments; leave unset for localhost http."""
    return os.environ.get("SOPPROOF_COOKIE_SECURE", "").strip().lower() in (
        "1", "true", "yes",
    )


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def dummy_verify(password: str) -> None:
    try:
        _hasher.verify(_DUMMY_HASH, password)
    except Exception:
        pass


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def new_raw_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(raw_token: str) -> str:
    return _token_hash(raw_token)


def start_session(user_id: int) -> str:
    store.cleanup_expired_sessions()
    raw = new_raw_token()
    store.create_session(_token_hash(raw), user_id, time.time() + session_days() * 86400)
    return raw


def guest_hours() -> int:
    try:
        return max(1, min(24, int(os.environ.get("SOPPROOF_GUEST_HOURS", "4"))))
    except ValueError:
        return 4


def start_guest_session() -> str:
    """Create a guest session. No user account is created or referenced."""
    store.cleanup_expired_sessions()
    raw = new_raw_token()
    store.create_guest_session(_token_hash(raw), time.time() + guest_hours() * 3600)
    return raw


GUEST_PRINCIPAL = {"id": None, "email": "guest", "is_admin": 0, "is_guest": True}


def is_guest(user) -> bool:
    return bool(user and user.get("is_guest"))


def set_session_cookies(response, raw_session: str, max_age_seconds: int = None) -> str:
    csrf = new_raw_token()
    max_age = max_age_seconds or session_days() * 86400
    response.set_cookie(
        SESSION_COOKIE, raw_session, max_age=max_age,
        httponly=True, samesite="lax", secure=cookie_secure(), path="/",
    )
    response.set_cookie(
        CSRF_COOKIE, csrf, max_age=max_age,
        httponly=False, samesite="lax", secure=cookie_secure(), path="/",
    )
    return csrf


def clear_session_cookies(response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


def current_user(request: Request):
    """Return the principal for a valid session cookie, else None.

    Normal principals are user rows with ``is_guest=False``. Guest
    principals are a fixed dict with ``id=None`` and ``is_guest=True`` —
    they own nothing and can only ever read the demo dataset.
    """
    raw = request.cookies.get(SESSION_COOKIE)
    if not raw:
        return None
    token = _token_hash(raw)
    session = store.get_session(token)
    if session:
        now = time.time()
        if session["expires_at"] < now:
            store.delete_session(token)
            return None
        # Sliding renewal: extend when past half the session lifetime.
        if session["expires_at"] - now < session_days() * 43200:
            store.touch_session(token, now + session_days() * 86400)
        else:
            store.touch_session(token, session["expires_at"])
        user = store.get_user_by_id(session["user_id"])
        if user is not None:
            user["is_guest"] = False
        return user
    guest = store.get_guest_session(token)
    if guest:
        if guest["expires_at"] < time.time():
            store.delete_guest_session(token)
            return None
        principal = dict(GUEST_PRINCIPAL)
        # Internal only: scopes guest credits/ownership to this session.
        # Never serialised by _public_user.
        principal["session_hash"] = token
        return principal
    return None


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_admin(request: Request):
    user = require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=404, detail="Not found.")
    return user


def require_csrf(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF validation failed.")


# ---------- minimal in-memory rate limiting ----------
_hits = defaultdict(list)


def rate_limit(key: str, limit: int, window_seconds: int) -> None:
    now = time.time()
    bucket = [t for t in _hits[key] if now - t < window_seconds]
    if len(bucket) >= limit:
        _hits[key] = bucket
        raise HTTPException(
            status_code=429, detail="Too many attempts. Please wait and retry."
        )
    bucket.append(now)
    _hits[key] = bucket


def client_key(request: Request, extra: str = "") -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{extra}"
