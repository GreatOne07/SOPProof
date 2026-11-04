# SOPProof — Azure App Service Deployment (B1, Linux, Python 3.11)

Status: deployment-ready after the accept-and-poll rework. No Azure
resources are created by this document; every step below is manual.

## 1. Architecture (unchanged pipeline)

One App Service app runs the whole product:

- FastAPI (`backend.app:app`) serves the SPA (`frontend/index.html`),
  the API, and ownership-checked file routes. No static mounts.
- SAM 3.1 and Muse Spark are **remote Meta API calls** — no GPU,
  no local model files, no FFmpeg binary (OpenCV headless wheels carry
  their own codecs; the video pipeline shells out to Python stage
  scripts with the same interpreter).
- SQLite (`data/sopproof.db` locally) is the auth/session/ownership/
  job index. **One worker, one instance** — SQLite, the in-memory
  rate limiter, and local evidence files all assume it. Do not scale
  out or add workers.

## 2. App Service configuration

| Setting | Value |
|---|---|
| Runtime stack | Python 3.11 (Linux) |
| Plan | B1 (Always On supported) |
| Startup command | `python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1` |
| Always On | On (B1) |
| Health check path | `/api/health` |
| Instances / workers | 1 / 1 — never increase |

App Settings (environment variables):

| Name | Value | Notes |
|---|---|---|
| `MODEL_API_KEY` | *(secret — set in portal only)* | Meta API key for SAM 3.1 + Muse Spark. Never commit it; `.env` is local-only and ignored by Git. |
| `SOPPROOF_COOKIE_SECURE` | `1` | Marks session cookies Secure behind Azure HTTPS. Required in production. |
| `SOPPROOF_DB_PATH` | `/home/sopproof-data/sopproof.db` | Persistent SQLite location. |
| `SOPPROOF_UPLOAD_DIR` | `/home/sopproof-data/uploads` | Persistent original uploads. |
| `SOPPROOF_OUTPUT_DIR` | `/home/sopproof-data/outputs` | Persistent reports/frames/previews. |
| `SOPPROOF_GUEST_HOURS` | `4` (default) | Optional. Guest session lifetime. |
| `SOPPROOF_SESSION_DAYS` | `7` (default) | Optional. User session lifetime. |

`/home` is the persistent filesystem on App Service Linux. Directories
are created by the app on startup. Local development keeps the current
defaults (`data/`, `uploads/`, `outputs/` inside the repo) whenever
these variables are unset.

## 3. How inspections run (accept-and-poll)

`POST /api/inspect/photo|video` validates the upload, consumes a guest
credit (guests only, atomically), stores the file, creates a job row,
and returns **202** with `inspection_id` and `status_url`. The existing
pipeline then runs unchanged in the background. Clients poll
`GET /api/inspections/{id}` (owner-only; 404 for anyone else) for
`pending → processing → completed | failed`. Completed responses carry
the same result payload the old synchronous POST returned. Jobs left
mid-flight by a restart are marked `failed` on startup. This design
exists because Azure's load balancer drops HTTP connections idle for
~230 s, which long video inspections would otherwise hit.

## 4. Deployment steps

1. Review `.gitignore` (excludes `.env`, `data/`, `uploads/`,
   `outputs/`, `.venv/`, backups, `*.bak`, large test videos). Never
   deploy the local `uploads/`, `outputs/`, or `data/` contents.
2. Confirm `requirements.txt` installs cleanly:
   `python -m venv .venv && .venv/bin/pip install -r requirements.txt`
   then `.venv/bin/python -m unittest discover -s backend -p "test_*.py"`.
3. Create the App Service (Python 3.11, B1, Linux) and set the startup
   command and App Settings from §2.
4. Deploy the repo contents (GitHub Actions, Azure CLI `az webapp up`,
   or ZIP deploy). The package must include `backend/`, `frontend/`,
   `demo_data/`, `hardhat_sample.jpg` (served at `/hero-photo.jpg`),
   and the root pipeline scripts `test_sam_video.py`,
   `test_sam_video_person.py`, `build_video_detections.py`.
5. Create the first admin once, via SSH into the app:
   `python -m backend.bootstrap_admin --email you@example.com`
   (interactive password; refuses to run after any user exists).
6. Invite users from the admin UI; registration is invite-only.

## 5. Post-deployment checks

- `GET https://<app>/api/health` → `{"status": "healthy", ...}`.
- Sign in as admin; upload a photo; confirm the UI polls to a result
  and the report opens; restart the app and confirm the report and
  evidence still load (persistence under `/home/sopproof-data`).
- As a guest: confirm "2 of 2 inspections remaining", run one photo
  inspection (counter → 1), exhaust the second, confirm the 403 limit
  message and disabled Run buttons; confirm demo replays stay free.
- Confirm a second user/guest cannot open the first user's report or
  job status URL (404).
- Watch the $10 budget alert: B1 alone is ~$13/mo at list price.

## 6. Limitations

- Single instance only (see §1). No horizontal scaling without
  replacing SQLite + local file storage.
- In-memory rate limiting resets on restart (per-process by design).
- Background jobs run in-process; a mid-inspection restart fails the
  job (marked `failed` at next startup) and guest credits already
  consumed are not refunded — this is stated in the UI.
- Video inspections still take minutes of wall-clock pipeline time;
  polling keeps the browser request short, but the work is unchanged.
- Azure Files is unnecessary while `/home` persistence suffices; if
  evidence outgrows the plan's disk quota, mount Azure Files and point
  the three `SOPPROOF_*_DIR` settings at the mount instead.
