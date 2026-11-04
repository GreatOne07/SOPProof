"""One-time migration: assign pre-authentication (legacy) inspection
records to the first administrator account.

Usage:
    .venv/bin/python -m backend.claim_legacy

Scans the same artifacts the report history already recognises (photo
report HTML, video job directories) plus orphaned uploads, and records
ownership for the first admin. Safe to rerun: ownership rows are keyed
by inspection_id and inserted with INSERT OR IGNORE, so nothing is
duplicated and existing owners are never overwritten.
"""
import re
import sys
from pathlib import Path

from backend import store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

_PHOTO_REPORT = re.compile(r"^([a-f0-9]{12})_report\.html$")
_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_UPLOAD = re.compile(r"^([a-f0-9]{12})\.(jpg|jpeg|png|webp|mp4)$")


def main() -> int:
    store.init_db()
    admin = store.first_admin()
    if not admin:
        print("No administrator account exists. Run backend.bootstrap_admin first.")
        return 1

    claimed = 0
    already = 0

    def claim(inspection_id: str, kind: str) -> None:
        nonlocal claimed, already
        if store.owner_of(inspection_id) is not None:
            already += 1
            return
        store.record_inspection(inspection_id, admin["id"], kind)
        claimed += 1

    for path in OUTPUT_DIR.glob("*_report.html"):
        match = _PHOTO_REPORT.match(path.name)
        if match:
            claim(match.group(1), "photo")

    for path in OUTPUT_DIR.glob("*/outputs/video_inspection_report.html"):
        job_id = path.parent.parent.name
        if _JOB_ID.match(job_id):
            claim(job_id, "video")

    for path in UPLOAD_DIR.iterdir():
        match = _UPLOAD.match(path.name)
        if match:
            claim(match.group(1), "video" if match.group(2) == "mp4" else "photo")

    print(f"Legacy claim complete for admin {admin['email']}: "
          f"{claimed} newly claimed, {already} already owned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
