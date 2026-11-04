"""Focused backend-hardening checks. No network/AI calls.

Run: .venv/bin/python -m unittest backend.test_hardening -v
"""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "SOPPROOF_DB_PATH", str(Path(tempfile.mkdtemp()) / "test-hardening.db")
)

from fastapi.testclient import TestClient

from backend import auth, store
from backend.app import app
from backend.video_inspection import select_sample_frames
from backend.video_pipeline import _safe_original_name


def _csrf(client: TestClient) -> str:
    return client.get("/api/auth/csrf").json()["csrf_token"]


def _post(client: TestClient, url, payload=None):
    return client.post(url, json=payload or {},
                       headers={"x-csrf-token": _csrf(client)})


class SampleFramesTests(unittest.TestCase):
    def test_exactly_five_spanning_duration(self):
        self.assertEqual(select_sample_frames(100), [0, 25, 50, 74, 99])
        frames = select_sample_frames(10000)
        self.assertEqual((len(frames), frames[0], frames[-1]), (5, 0, 9999))

    def test_short_and_empty(self):
        self.assertEqual(select_sample_frames(3), [0, 1, 2])
        self.assertEqual(select_sample_frames(5), [0, 1, 2, 3, 4])
        self.assertEqual(select_sample_frames(0), [])


class SafeOriginalNameTests(unittest.TestCase):
    def test_traversal_stripped_and_long_capped(self):
        self.assertEqual(_safe_original_name("../../etc/evil.mp4", "x.mp4"), "evil.mp4")
        self.assertEqual(_safe_original_name("", "x.mp4"), "x.mp4")
        self.assertLessEqual(len(_safe_original_name("a" * 300 + ".mp4", "x.mp4")), 100)


class OutputExposureTests(unittest.TestCase):
    def setUp(self):
        db = Path(os.environ["SOPPROOF_DB_PATH"])
        db.unlink(missing_ok=True)
        store.init_db()
        auth._hits.clear()
        store.create_user(
            "hardening@test.local",
            auth.hash_password("hardening-password-1"),
            is_admin=True,
        )
        from backend.claim_legacy import main as claim_legacy

        claim_legacy()
        self.client = TestClient(app, raise_server_exceptions=False)
        resp = _post(self.client, "/api/auth/login", {
            "email": "hardening@test.local",
            "password": "hardening-password-1",
        })
        assert resp.status_code == 200, resp.text

    def test_internal_and_legacy_outputs_blocked(self):
        for path in (
            "/outputs/video_inspection.json",
            "/outputs/video_inspection_timeline.txt",
            "/outputs/hardhat_report.html",
            "/outputs/954010f3b9fc/outputs/video_inspection_report.html.bak",
        ):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_reports_list_only_valid_records(self):
        anon = TestClient(app, raise_server_exceptions=False)
        self.assertEqual(anon.get("/api/reports").status_code, 401)
        data = self.client.get("/api/reports").json()
        for item in data["reports"]:
            url = item["url"]
            self.assertNotIn("hardhat_report", url)
            self.assertNotIn("video_inspection_report.html\"", url)
            parts = url.split("/")
            if url.endswith("_report.html") and len(parts) == 3:
                self.assertRegex(parts[2], r"^[a-f0-9]{12}_report\.html$")

    def test_upload_validation_and_health(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        anon = TestClient(app, raise_server_exceptions=False)
        unauthenticated = anon.post(
            "/api/inspect/photo", files={"file": ("evil.exe", b"x")}
        )
        self.assertEqual(unauthenticated.status_code, 401)
        bad = self.client.post(
            "/api/inspect/photo", files={"file": ("evil.exe", b"x")},
            headers={"x-csrf-token": _csrf(self.client)},
        )
        self.assertEqual(bad.status_code, 400)
        bad_video = self.client.post(
            "/api/inspect/video", files={"file": ("../evil.mov", b"x")},
            headers={"x-csrf-token": _csrf(self.client)},
        )
        self.assertEqual(bad_video.status_code, 400)


if __name__ == "__main__":
    unittest.main()
