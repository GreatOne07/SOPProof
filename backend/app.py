import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import mimetypes

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend import auth as auth_mod
from backend import store
from backend.inspection import inspect_image
from backend.report import generate_html_report
from backend.video_pipeline import run_video_pipeline


# --------------------------------------
# PROJECT DIRECTORIES
# --------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local development defaults; production (Azure) overrides both to
# persistent storage, e.g. /home/sopproof-data/uploads and .../outputs.
UPLOAD_DIR = Path(os.environ.get("SOPPROOF_UPLOAD_DIR", PROJECT_ROOT / "uploads"))
OUTPUT_DIR = Path(os.environ.get("SOPPROOF_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"
# Public marketing photo for the welcome hero. This is the repository's own
# unwatermarked sample photograph (hardhat_sample.jpg at the project root),
# not user evidence: it is not an upload, report or inspection record.
HERO_PHOTO_FILE = PROJECT_ROOT / "hardhat_sample.jpg"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE = 15 * 1024 * 1024

MAX_VIDEO_SIZE = 100 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------
# FASTAPI APPLICATION
# --------------------------------------

app = FastAPI(
    title="SOPProof API",
    description="AI-Assisted Workplace Safety Inspection",
    version="1.0.0",
)

store.init_db()

# NOTE: /outputs and /uploads are NOT static mounts. Every file is served
# through the ownership-checked routes below so one user can never fetch
# another user's reports, uploads, frames or previews.


# Internal job/debug files live under /outputs for convenience, but raw SAM
# responses, execution logs, CSVs and JSON state are not public report assets.
# Only required report assets are served: valid current report HTML,
# evidence images and report videos.
_ALLOWED_OUTPUT_SUFFIXES = {".html", ".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi"}
_VALID_PHOTO_REPORT = re.compile(r"^[a-f0-9]{12}_report\.html$")
_VALID_JOB_ID = re.compile(r"^[a-f0-9]{12}$")


@app.middleware("http")
async def block_internal_output_files(request, call_next):
    if request.url.path.startswith("/outputs/"):
        relative = request.url.path[len("/outputs/"):]
        parts = [p for p in relative.split("/") if p not in ("", ".")]
        suffix = Path(relative).suffix.lower()
        allowed = suffix in _ALLOWED_OUTPUT_SUFFIXES and ".." not in parts
        if allowed and suffix == ".html":
            allowed = bool(
                (len(parts) == 1 and _VALID_PHOTO_REPORT.match(parts[0]))
                or (
                    len(parts) == 3
                    and parts[1] == "outputs"
                    and parts[2] == "video_inspection_report.html"
                    and _VALID_JOB_ID.match(parts[0])
                )
            )
        if not allowed:
            return JSONResponse(
                status_code=404,
                content={"detail": "Not found."},
            )

    return await call_next(request)


# --------------------------------------
# DASHBOARD
# --------------------------------------

@app.get("/login", include_in_schema=False)
@app.get("/register", include_in_schema=False)
@app.get("/overview", include_in_schema=False)
@app.get("/photo", include_in_schema=False)
@app.get("/video", include_in_schema=False)
@app.get("/reports", include_in_schema=False)
@app.get("/reports/{report_id}", include_in_schema=False)
@app.get("/")
def dashboard():
    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend index.html not found.",
        )

    return FileResponse(FRONTEND_FILE)


@app.get("/hero-photo.jpg", include_in_schema=False)
def hero_photo():
    if not HERO_PHOTO_FILE.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(HERO_PHOTO_FILE, media_type="image/jpeg")


# --------------------------------------
# HEALTH CHECK
# --------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "SOPProof",
        "photo_inspection": "available",
        "video_inspection": "available",
        "report_generation": "available",
    }


# --------------------------------------
# AUTHENTICATION (invite-only)
# --------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str
    password: str


class RegisterBody(BaseModel):
    invite_token: str
    email: str
    password: str


class InviteBody(BaseModel):
    email: str | None = None
    expires_hours: int = 72


def _public_user(user):
    public = {
        "id": user["id"],
        "email": user["email"],
        "is_admin": bool(user.get("is_admin")),
        "is_guest": bool(user.get("is_guest")),
    }
    if public["is_guest"]:
        public["guest_inspections"] = _guest_allowance(user)
    return public


def _guest_allowance(user):
    """Current guest inspection allowance for a guest principal."""
    used = store.guest_credits_used(user.get("session_hash") or "")
    used = used if used is not None else store.GUEST_INSPECTION_LIMIT
    limit = store.GUEST_INSPECTION_LIMIT
    return {"limit": limit, "used": used, "remaining": max(0, limit - used)}


def _consume_guest_credit_or_403(user) -> None:
    """Consume one credit for a validated guest inspection, or raise 403.

    Called only after upload validation has passed, at the point the real
    inspection is accepted and begins. Credits are not refunded if the AI
    pipeline later fails.
    """
    if not store.consume_guest_credit(user.get("session_hash") or ""):
        raise HTTPException(
            status_code=403,
            detail=(
                "Demo inspection limit reached. You've used both inspections "
                "available in this guest session. Sign in to continue "
                "inspecting workplace evidence."
            ),
        )


def _record_inspection_owner(user, inspection_id: str, kind: str) -> None:
    if auth_mod.is_guest(user):
        store.record_guest_inspection(
            inspection_id, user.get("session_hash") or "", kind)
    else:
        store.record_inspection(inspection_id, user["id"], kind)


def _owner_key(user) -> str:
    """Stable job-ownership key for a user or guest principal."""
    if auth_mod.is_guest(user):
        return "guest:" + (user.get("session_hash") or "")
    return f"user:{user['id']}"


@app.on_event("startup")
def _sweep_orphaned_jobs() -> None:
    # Jobs left mid-flight by a previous process can never complete.
    store.fail_orphaned_jobs()


@app.get("/api/inspections/{inspection_id}")
def inspection_status(request: Request, inspection_id: str):
    """Poll an accepted inspection job. Owner-only; 404 for anyone else."""
    user = auth_mod.require_user(request)
    job = store.get_job(inspection_id)
    if not job or job["owner_key"] != _owner_key(user):
        raise HTTPException(status_code=404, detail="Not found.")
    payload = {
        "inspection_id": job["inspection_id"],
        "kind": job["kind"],
        "status": job["status"],
    }
    if job["status"] == "completed" and job["result_json"]:
        payload.update(json.loads(job["result_json"]))
    elif job["status"] == "failed":
        payload["error"] = job["error"] or "Inspection failed."
    return payload


def _validated_credentials(body: Credentials):
    email = (body.email or "").strip().lower()
    if not _EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if not body.password or len(body.password) < 12:
        raise HTTPException(status_code=400, detail="Password must be at least 12 characters.")
    if len(body.password) > 256:
        raise HTTPException(status_code=400, detail="Password is too long.")
    return email


@app.get("/api/auth/csrf")
def auth_csrf(request: Request):
    existing = request.cookies.get(auth_mod.CSRF_COOKIE)
    token = existing or auth_mod.new_raw_token()
    response = JSONResponse(content={"csrf_token": token})
    if not existing:
        response.set_cookie(
            auth_mod.CSRF_COOKIE, token, max_age=auth_mod.session_days() * 86400,
            httponly=False, samesite="lax", secure=auth_mod.cookie_secure(), path="/",
        )
    return response


@app.post("/api/auth/guest", status_code=201)
def auth_guest(request: Request):
    """Start a guest demo session with 2 real-inspection credits.

    No account is created. The session may browse the synthetic demo
    dataset, run up to the credited number of real inspections, and read
    only the reports those inspections create.
    """
    auth_mod.require_csrf(request)
    auth_mod.rate_limit(auth_mod.client_key(request, "guest"), 20, 3600)
    raw = auth_mod.start_guest_session()
    principal = dict(auth_mod.GUEST_PRINCIPAL)
    principal["session_hash"] = auth_mod.token_hash(raw)
    response = JSONResponse(
        status_code=201,
        content={"user": _public_user(principal)},
    )
    auth_mod.set_session_cookies(
        response, raw, max_age_seconds=auth_mod.guest_hours() * 3600
    )
    return response


@app.post("/api/auth/register")
def auth_register(request: Request, body: RegisterBody):
    auth_mod.require_csrf(request)
    if auth_mod.is_guest(auth_mod.current_user(request)):
        raise HTTPException(
            status_code=403,
            detail="Guest sessions cannot register accounts. Exit the demo and use an invitation link.",
        )
    auth_mod.rate_limit(auth_mod.client_key(request, "register"), 10, 300)
    email = _validated_credentials(body)
    invite = store.get_invitation_by_hash(auth_mod.token_hash(body.invite_token or ""))
    now = time.time()
    if (
        not invite
        or invite["used_at"] is not None
        or invite["revoked_at"] is not None
        or invite["expires_at"] < now
        or (invite.get("email") and invite["email"].lower() != email)
    ):
        raise HTTPException(status_code=400, detail="This invitation is invalid or expired.")
    if store.get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Registration could not be completed.")
    user_id = store.create_user(email, auth_mod.hash_password(body.password), is_admin=False)
    if not store.mark_invitation_used(invite["id"]):
        store.delete_user(user_id)
        raise HTTPException(status_code=400, detail="This invitation is invalid or expired.")
    user = store.get_user_by_id(user_id)
    raw = auth_mod.start_session(user_id)
    response = JSONResponse(status_code=201, content={"user": _public_user(user)})
    auth_mod.set_session_cookies(response, raw)
    return response


@app.post("/api/auth/login")
def auth_login(request: Request, body: Credentials):
    auth_mod.require_csrf(request)
    email = (body.email or "").strip().lower()
    auth_mod.rate_limit(auth_mod.client_key(request, "login:" + email), 10, 300)
    user = store.get_user_by_email(email) if _EMAIL_RE.match(email) else None
    if not user:
        auth_mod.dummy_verify(body.password or "")
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not auth_mod.verify_password(user["password_hash"], body.password or ""):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    raw = auth_mod.start_session(user["id"])
    response = JSONResponse(content={"user": _public_user(user)})
    auth_mod.set_session_cookies(response, raw)
    return response


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    user = auth_mod.require_user(request)
    auth_mod.require_csrf(request)
    raw = request.cookies.get(auth_mod.SESSION_COOKIE)
    if raw:
        store.delete_session(auth_mod.token_hash(raw))
        store.delete_guest_session(auth_mod.token_hash(raw))
    response = JSONResponse(content={"ok": True})
    auth_mod.clear_session_cookies(response)
    return response


@app.get("/api/auth/me")
def auth_me(request: Request):
    return {"user": _public_user(auth_mod.require_user(request))}


# --------------------------------------
# ADMIN: INVITATIONS
# --------------------------------------

@app.post("/api/admin/invitations", status_code=201)
def admin_create_invitation(request: Request, body: InviteBody):
    admin = auth_mod.require_admin(request)
    auth_mod.require_csrf(request)
    email = (body.email or "").strip().lower() or None
    if email and not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    hours = max(1, min(720, int(body.expires_hours or 72)))
    raw = auth_mod.new_raw_token()
    invite_id = store.create_invitation(
        auth_mod.token_hash(raw), email, admin["id"], time.time() + hours * 3600
    )
    # The raw token is returned exactly once, to the admin, so it can be
    # shared out-of-band. Only its hash is stored; it is never logged and
    # never included in invitation list responses.
    return {
        "id": invite_id,
        "email": email,
        "invite_token": raw,
        "register_path": f"/register?invite={raw}",
    }


@app.get("/api/admin/invitations")
def admin_list_invitations(request: Request):
    auth_mod.require_admin(request)
    now = time.time()
    out = []
    for inv in store.list_invitations():
        status = (
            "used" if inv["used_at"] is not None
            else "revoked" if inv["revoked_at"] is not None
            else "expired" if inv["expires_at"] < now
            else "pending"
        )
        out.append({
            "id": inv["id"], "email": inv["email"], "status": status,
            "created_at": inv["created_at"], "expires_at": inv["expires_at"],
        })
    return {"invitations": out}


@app.delete("/api/admin/invitations/{invitation_id}")
def admin_revoke_invitation(request: Request, invitation_id: int):
    auth_mod.require_admin(request)
    auth_mod.require_csrf(request)
    if not store.revoke_invitation(invitation_id):
        raise HTTPException(status_code=404, detail="Invitation not found.")
    return {"ok": True}


# --------------------------------------
# PROTECTED FILE ACCESS
# --------------------------------------

_UPLOAD_NAME_RE = re.compile(r"^([a-f0-9]{12})\.(jpg|jpeg|png|webp|mp4)$")

# Guest demo dataset: synthetic, physically separate from real records.
# Guests may read ONLY this directory; real users never see it listed.
DEMO_DIR = PROJECT_ROOT / "demo_data"


def _serve_file(target: Path):
    resolved = target.resolve()
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    media, _ = mimetypes.guess_type(str(resolved))
    # Private evidence must never sit in a shared/browser cache where the
    # next user of this browser could be served another user's files.
    return FileResponse(
        resolved,
        media_type=media or "application/octet-stream",
        headers={"Cache-Control": "private, no-store"},
    )


def _demo_target(parts):
    """Resolve a request path inside the demo dataset, or None."""
    if not parts:
        return None
    target = DEMO_DIR.joinpath(*parts)
    resolved = target.resolve()
    if DEMO_DIR.resolve() not in resolved.parents and resolved != DEMO_DIR.resolve():
        return None
    return resolved if resolved.is_file() else None


def _owned_file(user, inspection_id: str, target: Path):
    if not inspection_id or store.owner_of(inspection_id) != user["id"]:
        raise HTTPException(status_code=404, detail="Not found.")
    return _serve_file(target)


def _guest_owns(user, inspection_id: str) -> bool:
    """True when a guest session created this inspection in-session."""
    return bool(inspection_id) and store.guest_session_of(inspection_id) == (
        user.get("session_hash") or ""
    )


@app.get("/uploads/{name}", include_in_schema=False)
def serve_upload(request: Request, name: str):
    user = auth_mod.require_user(request)
    match = _UPLOAD_NAME_RE.match(name)
    if not match:
        raise HTTPException(status_code=404, detail="Not found.")
    if auth_mod.is_guest(user):
        # Guests may read their own in-session uploads and the synthetic
        # demo evidence — never another principal's uploads.
        if _guest_owns(user, match.group(1)):
            return _serve_file(UPLOAD_DIR / name)
        target = _demo_target([name])
        if target is None:
            raise HTTPException(status_code=404, detail="Not found.")
        return _serve_file(target)
    return _owned_file(user, match.group(1), UPLOAD_DIR / name)


@app.get("/outputs/{path:path}", include_in_schema=False)
def serve_output(request: Request, path: str):
    user = auth_mod.require_user(request)
    parts = [p for p in path.split("/") if p not in ("", ".")]
    suffix = Path(path).suffix.lower()
    allowed = suffix in _ALLOWED_OUTPUT_SUFFIXES and ".." not in parts
    if allowed and suffix == ".html":
        allowed = bool(
            (len(parts) == 1 and _VALID_PHOTO_REPORT.match(parts[0]))
            or (
                len(parts) == 3
                and parts[1] == "outputs"
                and parts[2] == "video_inspection_report.html"
                and _VALID_JOB_ID.match(parts[0])
            )
        )
    if not allowed or not parts:
        raise HTTPException(status_code=404, detail="Not found.")
    if len(parts) == 1 and _VALID_PHOTO_REPORT.match(parts[0]):
        inspection_id = parts[0][:-len("_report.html")]
    elif _VALID_JOB_ID.match(parts[0]):
        inspection_id = parts[0]
    else:
        raise HTTPException(status_code=404, detail="Not found.")
    if auth_mod.is_guest(user):
        # Guests may read reports/evidence from their own in-session
        # inspections and the synthetic demo dataset — nothing else.
        if _guest_owns(user, inspection_id):
            target = OUTPUT_DIR.joinpath(*parts)
            resolved = target.resolve()
            if OUTPUT_DIR.resolve() not in resolved.parents and resolved != OUTPUT_DIR.resolve():
                raise HTTPException(status_code=404, detail="Not found.")
            return _serve_file(target)
        target = _demo_target(parts)
        if target is None:
            raise HTTPException(status_code=404, detail="Not found.")
        return _serve_file(target)
    target = OUTPUT_DIR.joinpath(*parts)
    if OUTPUT_DIR.resolve() not in target.resolve().parents and target.resolve() != OUTPUT_DIR.resolve():
        raise HTTPException(status_code=404, detail="Not found.")
    return _owned_file(user, inspection_id, target)


# --------------------------------------
# PHOTO INSPECTION
# --------------------------------------

def _run_photo_job(user, inspection_id: str, image_path: Path) -> None:
    """Background photo pipeline. Runs the unchanged inspection engine;
    the outcome is recorded on the job row for owner-only polling."""
    safe_name = image_path.name
    store.update_job(inspection_id, "processing")
    try:
        report = inspect_image(image_path)
        if not isinstance(report, dict):
            raise RuntimeError("Inspection engine returned an invalid report.")
        html_path = OUTPUT_DIR / f"{inspection_id}_report.html"
        generate_html_report(report, html_path)
        if not html_path.is_file():
            raise RuntimeError("Photo report was not generated.")
        _record_inspection_owner(user, inspection_id, "photo")
        result = {
            "success": True,
            "inspection_id": inspection_id,
            "source_image": safe_name,
            "overall_result": report.get("overall_result", "uncertain"),
            "human_review_required": report.get("human_review_required", True),
            "people": report.get("people", []),
            "image_url": f"/uploads/{safe_name}",
            "report_url": f"/outputs/{html_path.name}",
            "message": "Photo inspection completed successfully.",
            **(
                {"guest_inspections": _guest_allowance(user)}
                if auth_mod.is_guest(user) else {}
            ),
        }
        store.update_job(inspection_id, "completed",
                         result_json=json.dumps(result))
    except Exception as exc:
        # Log details server-side only; clients get a generic failure.
        print(f"PHOTO INSPECTION FAILED [{inspection_id}]:", repr(exc))
        image_path.unlink(missing_ok=True)
        (OUTPUT_DIR / f"{inspection_id}_inspection.json").unlink(missing_ok=True)
        (OUTPUT_DIR / f"{inspection_id}_report.html").unlink(missing_ok=True)
        store.update_job(
            inspection_id, "failed",
            error="Photo inspection failed. Please retry or contact support.")


@app.post("/api/inspect/photo", status_code=202)
async def inspect_photo(request: Request, background_tasks: BackgroundTasks,
                        file: UploadFile = File(...)):
    user = auth_mod.require_user(request)
    auth_mod.require_csrf(request)
    auth_mod.rate_limit(
        f"inspect:{user.get('session_hash') or user['id']}", 30, 3600)

    original_name = (file.filename or "").replace("\\", "/")
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, PNG or WEBP image.",
        )

    image_data = await file.read(MAX_IMAGE_SIZE + 1)

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_data) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 15 MB.",
        )

    image_array = np.frombuffer(
        image_data,
        dtype=np.uint8,
    )

    decoded_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if decoded_image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    # Validation has passed: a guest inspection accepted here consumes
    # one credit atomically before processing begins (no refund on
    # later pipeline failure). Invalid uploads never reach this point.
    if auth_mod.is_guest(user):
        _consume_guest_credit_or_403(user)

    inspection_id = uuid.uuid4().hex[:12]
    safe_name = f"{inspection_id}{extension}"
    image_path = UPLOAD_DIR / safe_name

    image_path.write_bytes(image_data)

    store.create_job(inspection_id, _owner_key(user), "photo")
    background_tasks.add_task(
        _run_photo_job, dict(user), inspection_id, image_path)

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "inspection_id": inspection_id,
            "status": "pending",
            "status_url": f"/api/inspections/{inspection_id}",
            "message": "Photo inspection accepted and processing.",
            **(
                {"guest_inspections": _guest_allowance(user)}
                if auth_mod.is_guest(user) else {}
            ),
        },
    )



# --------------------------------------
# VIDEO INSPECTION
# --------------------------------------

def _run_video_job(user, inspection_id: str, video_path: Path,
                   original_name: str, safe_name: str) -> None:
    """Background video pipeline. Runs the unchanged staged pipeline;
    the outcome is recorded on the job row for owner-only polling."""
    store.update_job(inspection_id, "processing")
    try:
        report_path = run_video_pipeline(
            video_path, inspection_id, Path(original_name).name)
        if not report_path.is_file():
            raise RuntimeError("Video report was not generated.")
        json_path = (
            OUTPUT_DIR / inspection_id / "outputs" / "video_inspection.json")
        report_data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(report_data, dict):
            raise RuntimeError("Video inspection returned invalid data.")
        # Never report fake success when every Muse frame review failed.
        if not report_data.get("successful_frame_reviews", 0):
            raise RuntimeError(
                "Video review failed for all sampled frames. Please retry.")
        _record_inspection_owner(user, inspection_id, "video")
        result = {
            "success": True,
            "inspection_id": inspection_id,
            "source_video": safe_name,
            "overall_assessment": report_data.get(
                "overall_assessment", "Review completed."),
            "human_review_required": True,
            "successful_frame_reviews": report_data.get(
                "successful_frame_reviews", 0),
            "failed_frame_reviews": report_data.get(
                "failed_frame_reviews", 0),
            "report_url": (
                f"/outputs/{inspection_id}"
                "/outputs/video_inspection_report.html"),
            "message": "Video inspection completed successfully.",
            **(
                {"guest_inspections": _guest_allowance(user)}
                if auth_mod.is_guest(user) else {}
            ),
        }
        store.update_job(inspection_id, "completed",
                         result_json=json.dumps(result))
    except Exception as exc:
        print(f"VIDEO INSPECTION FAILED [{inspection_id}]:", repr(exc))
        video_path.unlink(missing_ok=True)
        shutil.rmtree(OUTPUT_DIR / inspection_id, ignore_errors=True)
        store.update_job(
            inspection_id, "failed",
            error="Video inspection failed. Please retry or contact support.")


@app.post("/api/inspect/video", status_code=202)
async def inspect_video(request: Request, background_tasks: BackgroundTasks,
                        file: UploadFile = File(...)):
    user = auth_mod.require_user(request)
    auth_mod.require_csrf(request)
    auth_mod.rate_limit(
        f"inspect:{user.get('session_hash') or user['id']}", 30, 3600)

    original_name = (file.filename or "").replace("\\", "/")
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4 video.",
        )

    video_data = await file.read(MAX_VIDEO_SIZE + 1)

    if not video_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    if len(video_data) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Video must be smaller than 100 MB.",
        )

    inspection_id = uuid.uuid4().hex[:12]
    safe_name = f"{inspection_id}.mp4"
    video_path = UPLOAD_DIR / safe_name
    video_path.write_bytes(video_data)

    # Verify that OpenCV can actually decode the uploaded video.
    cap = cv2.VideoCapture(str(video_path))
    valid_video = cap.isOpened()

    if valid_video:
        valid_video, _ = cap.read()

    cap.release()

    if not valid_video:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable MP4 video.",
        )

    # Decode validation passed: consume one guest credit atomically as
    # the real inspection is accepted (no refund on later failure).
    if auth_mod.is_guest(user):
        try:
            _consume_guest_credit_or_403(user)
        except HTTPException:
            video_path.unlink(missing_ok=True)
            raise

    store.create_job(inspection_id, _owner_key(user), "video")
    background_tasks.add_task(
        _run_video_job, dict(user), inspection_id, video_path,
        original_name, safe_name)

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "inspection_id": inspection_id,
            "status": "pending",
            "status_url": f"/api/inspections/{inspection_id}",
            "message": "Video inspection accepted and processing.",
            **(
                {"guest_inspections": _guest_allowance(user)}
                if auth_mod.is_guest(user) else {}
            ),
        },
    )


# --------------------------------------
# REPORT HISTORY
# --------------------------------------

# Only records generated by the current system (patterns defined above):
# photo reports are {12-hex}_report.html and video reports live inside a
# {12-hex} job directory. Legacy files are excluded, not deleted.
@app.get("/api/reports")
def list_reports(request: Request):
    user = auth_mod.require_user(request)

    if auth_mod.is_guest(user):
        # Guests see the synthetic demo dataset plus reports from their
        # own in-session inspections — never another principal's records.
        demo_files = [
            p for p in DEMO_DIR.glob("*_report.html")
            if _VALID_PHOTO_REPORT.match(p.name)
        ] + [
            p for p in DEMO_DIR.glob("*/outputs/video_inspection_report.html")
            if _VALID_JOB_ID.match(p.parent.parent.name)
        ]
        reports = [
            {
                "name": p.name,
                "url": "/outputs/" + p.relative_to(DEMO_DIR).as_posix(),
                "created": p.stat().st_mtime,
            }
            for p in demo_files
        ]
        own_ids = store.guest_inspection_ids(user.get("session_hash") or "")
        if own_ids:
            own_files = [
                p for p in OUTPUT_DIR.glob("*_report.html")
                if _VALID_PHOTO_REPORT.match(p.name)
                and p.name[: -len("_report.html")] in own_ids
            ] + [
                p for p in OUTPUT_DIR.glob("*/outputs/video_inspection_report.html")
                if _VALID_JOB_ID.match(p.parent.parent.name)
                and p.parent.parent.name in own_ids
            ]
            reports += [
                {
                    "name": p.name,
                    "url": "/outputs/" + p.relative_to(OUTPUT_DIR).as_posix(),
                    "created": p.stat().st_mtime,
                }
                for p in own_files
            ]
        reports.sort(key=lambda r: r["created"], reverse=True)
        return {"total": len(reports), "reports": reports, "demo": True}

    owned_ids = store.inspection_ids_for(user["id"])

    reports = []

    candidates = [
        path
        for path in OUTPUT_DIR.glob("*_report.html")
        if _VALID_PHOTO_REPORT.match(path.name)
    ] + [
        path
        for path in OUTPUT_DIR.glob("*/outputs/video_inspection_report.html")
        if _VALID_JOB_ID.match(path.parent.parent.name)
    ]

    def _report_id(path: Path) -> str:
        if path.name == "video_inspection_report.html":
            return path.parent.parent.name
        return path.name[: -len("_report.html")]

    report_files = sorted(
        (p for p in candidates if _report_id(p) in owned_ids),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for report_file in report_files:
        relative_url = report_file.relative_to(OUTPUT_DIR).as_posix()

        reports.append({
            "name": report_file.name,
            "url": f"/outputs/{relative_url}",
            "created": report_file.stat().st_mtime,
        })

    return {
        "total": len(reports),
        "reports": reports,
    }
