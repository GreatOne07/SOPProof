"""Guest demo mode security tests. No network/AI calls.

Run: .venv/bin/python -m unittest backend.test_guest -v
"""
import glob
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ["SOPPROOF_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test-guest.db")

from fastapi.testclient import TestClient

from backend import auth, store
from backend.app import DEMO_DIR, OUTPUT_DIR, UPLOAD_DIR, app

ADMIN_EMAIL = "admin@guest-test.local"
ADMIN_PW = "admin-password-123"
FAKE_ID = "abcdef123456"


def csrf(client: TestClient) -> str:
    return client.get("/api/auth/csrf").json()["csrf_token"]


def post(client, url, payload=None):
    return client.post(url, json=payload or {}, headers={"x-csrf-token": csrf(client)})


class GuestTestBase(unittest.TestCase):
    def setUp(self):
        Path(os.environ["SOPPROOF_DB_PATH"]).unlink(missing_ok=True)
        store.init_db()
        auth._hits.clear()
        store.create_user(ADMIN_EMAIL, auth.hash_password(ADMIN_PW), is_admin=True)
        self.admin = TestClient(app, raise_server_exceptions=False)
        assert post(self.admin, "/api/auth/login",
                    {"email": ADMIN_EMAIL, "password": ADMIN_PW}).status_code == 200
        self.guest = TestClient(app, raise_server_exceptions=False)
        resp = post(self.guest, "/api/auth/guest")
        assert resp.status_code == 201, resp.text

    def tearDown(self):
        (OUTPUT_DIR / f"{FAKE_ID}_report.html").unlink(missing_ok=True)
        (UPLOAD_DIR / f"{FAKE_ID}.jpg").unlink(missing_ok=True)

    def register_user(self, email):
        token = post(self.admin, "/api/admin/invitations",
                     {"email": email}).json()["invite_token"]
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/register",
                    {"invite_token": token, "email": email,
                     "password": "user-password-123"})
        assert resp.status_code == 201, resp.text
        return client


class GuestAccessTests(GuestTestBase):
    def test_1_guest_enters_without_registration_or_account(self):
        me = self.guest.get("/api/auth/me").json()["user"]
        self.assertTrue(me["is_guest"])
        self.assertFalse(me["is_admin"])
        self.assertIsNone(me["id"])
        self.assertEqual(store.count_users(), 1)  # only the seeded admin

    def test_2_3_guest_sees_only_demo_dashboard_data(self):
        data = self.guest.get("/api/reports").json()
        self.assertTrue(data.get("demo"))
        self.assertGreaterEqual(data["total"], 1)
        for item in data["reports"]:
            self.assertIn("de00000000", item["url"])
            resp = self.guest.get(item["url"])
            self.assertEqual(resp.status_code, 200, item["url"])
            self.assertIn("DEMO DATA", resp.text)

    def test_3b_guest_can_view_demo_evidence(self):
        self.assertEqual(self.guest.get("/uploads/de0000000009.jpg").status_code, 200)
        self.assertEqual(
            self.guest.get(
                "/outputs/de0000000003/outputs/video_review_frames/frame_0000.jpg"
            ).status_code, 200)
        self.assertEqual(
            self.guest.get("/outputs/de0000000003/demo-video.mp4").status_code, 200)

    def test_4_5_6_guest_cannot_access_real_reports_or_uploads(self):
        real_reports = sorted(glob.glob(str(OUTPUT_DIR / "*_report.html")))
        if real_reports:
            name = Path(real_reports[0]).name
            self.assertEqual(self.guest.get(f"/outputs/{name}").status_code, 404)
        real_uploads = sorted(glob.glob(str(UPLOAD_DIR / "*.*")))
        if real_uploads:
            name = Path(real_uploads[0]).name
            self.assertEqual(self.guest.get(f"/uploads/{name}").status_code, 404)
        # Normal user's synthetic private report is also invisible.
        user = self.register_user("owner@guest-test.local")
        uid = store.get_user_by_email("owner@guest-test.local")["id"]
        (OUTPUT_DIR / f"{FAKE_ID}_report.html").write_text("<html>private</html>")
        (UPLOAD_DIR / f"{FAKE_ID}.jpg").write_bytes(b"\xff\xd8private")
        store.record_inspection(FAKE_ID, uid, "photo")
        self.assertEqual(user.get(f"/outputs/{FAKE_ID}_report.html").status_code, 200)
        self.assertEqual(self.guest.get(f"/outputs/{FAKE_ID}_report.html").status_code, 404)
        self.assertEqual(self.guest.get(f"/uploads/{FAKE_ID}.jpg").status_code, 404)

    def test_7_8_9_guest_inspection_validation_never_consumes_credits(self):
        # Updated for the judge-evaluation allowance: guests may run real
        # inspections, but invalid uploads are rejected by validation
        # (400) before any credit is consumed or pipeline code can run.
        photo = self.guest.post(
            "/api/inspect/photo", files={"file": ("evil.exe", b"x")},
            headers={"x-csrf-token": csrf(self.guest)})
        self.assertEqual(photo.status_code, 400)
        video = self.guest.post(
            "/api/inspect/video", files={"file": ("evil.mov", b"x")},
            headers={"x-csrf-token": csrf(self.guest)})
        self.assertEqual(video.status_code, 400)
        # An undecodable "JPEG" is also a validation failure (400).
        photo2 = self.guest.post(
            "/api/inspect/photo", files={"file": ("site.jpg", b"\xff\xd8\xff")},
            headers={"x-csrf-token": csrf(self.guest)})
        self.assertEqual(photo2.status_code, 400)
        me = self.guest.get("/api/auth/me").json()["user"]
        self.assertEqual(me["guest_inspections"]["remaining"], 2)

    def test_10_11_guest_cannot_invite_register_or_modify(self):
        self.assertEqual(post(self.guest, "/api/admin/invitations").status_code, 404)
        self.assertEqual(self.guest.get("/api/admin/invitations").status_code, 404)
        resp = post(self.guest, "/api/auth/register",
                    {"invite_token": "x", "email": "g2@guest-test.local",
                     "password": "user-password-123"})
        self.assertEqual(resp.status_code, 403)
        # Review decisions are browser-local state in this app; there is no
        # backend write endpoint for guests (or anyone) to modify. The
        # inspections/reports write surface is covered by tests 7-10.

    def test_12_guest_cannot_escalate(self):
        # Forged/unknown session tokens are simply anonymous.
        forged = TestClient(app, raise_server_exceptions=False)
        forged.cookies.set("sopproof_session", "forged-admin-token")
        self.assertEqual(forged.get("/api/auth/me").status_code, 401)
        self.assertEqual(forged.get("/api/reports").status_code, 401)
        # Guest principal never carries admin rights anywhere.
        self.assertEqual(
            self.guest.delete("/api/admin/invitations/1",
                              headers={"x-csrf-token": csrf(self.guest)}).status_code,
            404)

    def test_13_guest_session_expires(self):
        store.create_guest_session(
            auth.token_hash("expired-guest-token"), time.time() - 5)
        client = TestClient(app, raise_server_exceptions=False)
        client.cookies.set("sopproof_session", "expired-guest-token")
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_14_15_admin_user_login_and_user_isolation_still_work(self):
        user_a = self.register_user("iso-a@guest-test.local")
        user_b = self.register_user("iso-b@guest-test.local")
        a_id = store.get_user_by_email("iso-a@guest-test.local")["id"]
        (OUTPUT_DIR / f"{FAKE_ID}_report.html").write_text("<html>A</html>")
        store.record_inspection(FAKE_ID, a_id, "photo")
        self.assertEqual(user_a.get(f"/outputs/{FAKE_ID}_report.html").status_code, 200)
        self.assertEqual(user_b.get(f"/outputs/{FAKE_ID}_report.html").status_code, 404)
        self.assertEqual(self.admin.get("/api/auth/me").json()["user"]["email"], ADMIN_EMAIL)
        # Real users never see the demo dataset in their list.
        admin_urls = [r["url"] for r in self.admin.get("/api/reports").json()["reports"]]
        self.assertFalse(any("de00000000" in u for u in admin_urls))
        self.assertEqual(
            self.admin.get("/outputs/de0000000001_report.html").status_code, 404)

    def test_demo_files_exist_and_are_labelled(self):
        self.assertTrue((DEMO_DIR / "de0000000001_report.html").is_file())
        html = (DEMO_DIR / "de0000000001_report.html").read_text(encoding="utf-8")
        self.assertIn("DEMO DATA", html)

    def test_guest_review_controls_are_local_demo_wired(self):
        # Regression: guest Human Verification controls must be interactive
        # demo controls (browser-local, session-scoped), never disabled and
        # never backed by a server write. Guards the served frontend only.
        frontend = (Path(__file__).resolve().parents[1]
                    / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn(
            "DEMO ACTION — saved locally only; "
            "not an official inspection decision.", frontend)
        self.assertNotIn("review decisions are disabled", frontend)
        self.assertNotIn("el.disabled=guest", frontend)
        self.assertIn("sopproof-demo-review-v1:", frontend)
        self.assertIn("clearDemoReviewState", frontend)


if __name__ == "__main__":
    unittest.main()
