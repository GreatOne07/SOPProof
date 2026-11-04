# SOPProof — AI Workplace Safety Inspection

**SOPProof turns workplace photos and videos into reviewable PPE evidence.** An inspector uploads an image or an MP4, Meta's SAM 3.1 detects workers and hard hats, Muse Spark visually reviews the original evidence worker by worker, and every potential concern is queued for a human decision. No verdict without proof: AI output alone is never treated as a confirmed violation.

Built with Meta AI models (`sam-3.1` + `muse-spark-1.3`) for a Meta AI hackathon context — see [Acknowledgements](#acknowledgements).

## Table of contents

- [The problem](#the-problem)
- [The solution](#the-solution)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Running the app](#running-the-app)
- [Project structure](#project-structure)
- [Limitations and future improvements](#limitations-and-future-improvements)
- [Security and privacy](#security-and-privacy)
- [Acknowledgements](#acknowledgements)

## The problem

Workplace safety inspections still run on clipboards, memory, and photos nobody re-checks:

- **Coverage is thin.** A safety officer cannot be on every site, every shift. Most PPE non-compliance is simply never observed.
- **Evidence is messy.** Inspection photos pile up in chat threads and shared drives with no link between an image, a finding, and a decision.
- **Review is inconsistent.** Whether a missing helmet gets flagged depends on who looked, how tired they were, and whether the photo survived.
- **Video is effectively unreviewable.** Hours of site footage go unwatched, so timestamped evidence of unsafe practice is lost.

## The solution

SOPProof inserts an AI evidence pipeline between *capture* and *decision*:

1. An inspector uploads a workplace **photo** (JPG/PNG/WebP) or **video** (MP4).
2. **SAM 3.1** detects people and hard hats in the original media.
3. **Muse Spark** visually reviews the evidence worker by worker and records helmet observations with plain-language evidence.
4. SOPProof generates a timestamped **HTML evidence report** and routes every potential concern to a **human verification** queue — confirm or dismiss, with an inspector note. Decisions stay with people; the AI only prepares the proof.

A read-only **Guest Demo** with precomputed synthetic reports — plus **2 live AI inspections per guest session** — lets anyone (for example, hackathon judges) evaluate the real pipeline without an account.

## Key features

| Feature | What it does |
|---|---|
| Photo inspection | Single-image PPE verification: workers and hard hats detected, each worker assessed, concerns flagged for review. |
| Video inspection | MP4 analysis across sampled frames, with a timestamped report and flagged-frame review (for example, "Frame 50 · 1.67s — Uncertain"). |
| SAM 3.1 detection | Meta SAM 3.1 (`sam-3.1`) finds people and hard hats and returns bounding boxes (and one-bit masks used internally to associate helmets with workers). |
| Muse Spark review | Muse Spark (`muse-spark-1.3`) visually reviews the original evidence per worker/frame and writes the assessment and evidence text. |
| Evidence reports | Self-contained HTML reports per inspection, an evidence library with search/filter/sort, and CSV export of selected reports. |
| Human verification | Confirm violation / dismiss as false positive, inspector notes, per-report review state stored in the browser. |
| Auth & ownership | Invite-only registration, Argon2-hashed passwords, server-side sessions. Every report, upload, frame, and job is ownership-checked — non-owners get 404. |
| Guest Demo | Opaque 4-hour guest sessions over clearly labelled synthetic demo data, isolated from all real records — plus exactly **2 real inspection credits** per guest session (photo and video share one allowance, enforced atomically by the backend). |

What SOPProof deliberately does **not** claim: saved reports store assessment text and frame stills, not SAM geometry — so the UI never draws fabricated boxes or masks, and there are no persistent SAM tracking IDs or pixel-mask visualisations. Only **people** and **hard hats** are detected today; other PPE classes shown in the UI are marked *coming soon* and are not connected to the inspection service.

## How it works

```
Upload (photo / MP4)
  → validation (type, size, decode)            [guest credit consumed here, once]
  → 202 Accepted + job ID ──► background pipeline ──► browser polls job status
        │
        ├─ Photo:  SAM 3.1 (people + hard hats) → Muse Spark per-worker review
        │          → HTML report
        └─ Video:  frame sampling → SAM 3.1 per frame → helmet↔worker association
                   → Muse Spark review of flagged frames → timestamped HTML report
  → Evidence library (owner-only) → Human verification (confirm / dismiss + note)
```

- **Photo flow** — `backend/inspection.py` sends the image to SAM 3.1 with *person* and *hard hat* prompts, matches helmets to workers by geometry (`backend/association.py`), then asks Muse Spark to review the original image worker by worker with those detections as context. `backend/report.py` renders the HTML report.
- **Video flow** — `backend/video_pipeline.py` runs staged subprocesses: SAM detection per sampled frame (`test_sam_video.py`, `test_sam_video_person.py`), detection assembly (`build_video_detections.py`), then Muse Spark frame review (`backend/run_video_review.py`). Only frames with successful reviews count; if every frame review fails, the inspection is reported as failed — never as a fake success.
- **Accept-and-poll** — inspections can run for minutes, so POSTs return `202` immediately and execute in the background (one worker, in-process). The frontend polls `GET /api/inspections/{id}`; jobs are database-backed and owner-checked, and jobs interrupted by a restart are marked failed rather than lost.

## Technology stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn (single worker) |
| AI models | Meta SAM 3.1 (`sam-3.1`) and Muse Spark (`muse-spark-1.3`) via `https://api.meta.ai/v1` (OpenAI Python SDK) |
| Vision utilities | OpenCV (headless), NumPy, Pillow |
| Storage | SQLite (users, sessions, ownership, guest credits, jobs); local filesystem for uploads/reports |
| Auth | Argon2 password hashing, server-side session cookies, CSRF double-submit tokens |
| Frontend | Single-file HTML/CSS/vanilla JS (`frontend/index.html`), dark + light themes |
| Tests | Python `unittest` (47 backend tests; SAM/Muse mocked — no API spend) |

## Screenshots

> 📷 **Screenshot placeholder — Overview dashboard.** Drop an image at `docs/screenshots/overview.png` and link it here: hero panel, latest-evidence card, inspection counters.

> 📷 **Screenshot placeholder — Photo inspection.** Drop an image at `docs/screenshots/photo-inspection.png`: uploaded evidence with per-worker findings ("Helmet Visible" cards).

> 📷 **Screenshot placeholder — Video inspection.** Drop an image at `docs/screenshots/video-inspection.png`: reviewed timestamps with flagged frames ("1.67s · FLAGGED").

> 📷 **Screenshot placeholder — Evidence library & human verification.** Drop an image at `docs/screenshots/evidence-library.png`: report table, filters, and the confirm/dismiss panel.

## Installation

Tested on macOS and Linux. You need Python 3.11+ and a Meta API key.

```bash
git clone https://github.com/GreatOne07/SOPProof.git
cd SOPProof

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` in the repository root (never commit it — it is Git-ignored):

```bash
MODEL_API_KEY=your-meta-api-key-here
```

Create the first admin account (interactive password prompt; refuses to run once any user exists):

```bash
python -m backend.bootstrap_admin --email you@example.com
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `MODEL_API_KEY` | Yes | Meta API key for SAM 3.1 + Muse Spark. Read from the environment (or local `.env`); never commit real values. |
| `SOPPROOF_DB_PATH` | No | SQLite location. Default: `data/sopproof.db`. |
| `SOPPROOF_UPLOAD_DIR` | No | Original-upload storage. Default: `uploads/`. |
| `SOPPROOF_OUTPUT_DIR` | No | Reports/frames storage. Default: `outputs/`. |
| `SOPPROOF_COOKIE_SECURE` | Production | Set to `1` behind HTTPS so session cookies are Secure. |
| `SOPPROOF_SESSION_DAYS` | No | User session lifetime, 1–30 days (default 7). |
| `SOPPROOF_GUEST_HOURS` | No | Guest session lifetime, 1–24 hours (default 4). |

Two further variables (`SOPPROOF_VIDEO_JOB_DIR`, `SOPPROOF_ORIGINAL_VIDEO_NAME`) are set internally between pipeline stages — you never configure them.

## Running the app

```bash
source .venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** in your browser. Sign in with the admin account, invite users from the admin panel, or choose **Explore as Guest** to try the demo (2 live inspections included). Health check: `GET /api/health`.

Run the test suite (mocked AI, no API credits used):

```bash
python -m unittest discover -s backend -p "test_*.py"
```

Deploying to Azure App Service (B1)? See [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## Project structure

```
SOPProof/
├── backend/
│   ├── app.py               # FastAPI app: auth, inspections (accept-and-poll), file serving
│   ├── auth.py              # Sessions, CSRF, rate limiting, guest sessions
│   ├── store.py             # SQLite: users, sessions, ownership, guest credits, jobs
│   ├── inspection.py        # Photo pipeline: SAM 3.1 + Muse Spark
│   ├── sam_detector.py      # SAM 3.1 client and box extraction
│   ├── meta_client.py       # Meta API client (MODEL_API_KEY)
│   ├── video_pipeline.py    # Staged video pipeline orchestration
│   ├── video_inspection.py  # Per-frame Muse Spark review
│   ├── report.py            # Photo HTML report generator
│   ├── video_report.py      # Video HTML report generator
│   ├── run_video_review.py  # Video review stage entry point
│   ├── bootstrap_admin.py   # One-time first-admin creation
│   ├── claim_legacy.py      # Claims pre-ownership legacy records for an admin
│   └── test_*.py            # Backend test suites
├── frontend/
│   └── index.html           # Entire single-file web app
├── demo_data/               # Synthetic guest-demo dataset (clearly labelled)
├── test_sam_video.py        # Video pipeline stage: frame detection (invoked at runtime)
├── test_sam_video_person.py # Video pipeline stage: person detection (invoked at runtime)
├── build_video_detections.py# Video pipeline stage: detection assembly (invoked at runtime)
├── hardhat_sample.jpg       # Public sample photo (welcome hero)
├── requirements.txt
└── AZURE_DEPLOYMENT.md
```

Runtime data (`data/`, `uploads/`, `outputs/`, `.env`, `.venv/`) is Git-ignored and never part of the repository.

## Limitations and future improvements

**Current limitations (implemented behaviour):**

- Single instance / single worker by design (SQLite + local files + in-memory rate limiting). No horizontal scaling.
- Background jobs run in-process; a mid-inspection restart marks the job failed (guest credits already consumed are not refunded — stated in the UI).
- Detection covers people and hard hats only. High-vis vests, fall harnesses, and restricted zones appear in the UI as *coming soon* and are not detected.
- Reports store assessment text and frame stills — not SAM boxes/masks — so no overlay or mask visualisation is drawn from saved data.
- Uploads: photos up to 15 MB (JPG/PNG/WebP); video MP4 only, up to 100 MB.
- Guest sessions expire after 4 hours; guest evidence then becomes inaccessible.

**Planned (not implemented):** additional PPE classes, a managed database/object-storage backend for multi-instance deployment, a durable job queue, and server-side audit logging of verification decisions.

## Security and privacy

- Passwords are Argon2-hashed; sessions are opaque server-side tokens in HttpOnly cookies (Secure behind HTTPS via `SOPPROOF_COOKIE_SECURE`).
- Every report, upload, frame, preview, download, and job-status request is ownership-checked per user or guest session; anything else returns 404, and expired guest sessions lose access to their evidence.
- Upload validation (extension, size, decode) runs before any inspection or guest-credit consumption; pipeline failures return generic messages — upstream errors, paths, and stack traces are logged server-side only.
- API keys come from the environment; no credentials, databases, or real user evidence are in this repository. The committed demo dataset is synthetic (one AI-generated demo photo, used with baked-in DEMO labels) and contains no real worker uploads.

## Acknowledgements

- **Meta AI** — SOPProof is built entirely on Meta's models: **SAM 3.1** for promptable people/hard-hat detection and **Muse Spark** for visual evidence review, both through the Meta API. This project was created in a Meta AI hackathon context to explore how foundation vision models can support — never replace — human safety judgement.
- The SAM 3.1 detection output parsing uses the `meta-sam-parser` package.
- Workplace sample imagery: the repository's own demonstration photographs and clearly labelled synthetic demo data.
