"""Guest real-inspection allowance tests (judge evaluation).

SAM 3.1 / Muse Spark are mocked at the app boundary — no API credits
are spent. Run: .venv/bin/python -m unittest backend.test_guest_inspections -v
"""
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ["SOPPROOF_DB_PATH"] = str(
    Path(tempfile.mkdtemp()) / "test-guest-inspections.db")

from fastapi.testclient import TestClient

import backend.app as app_mod
from backend import auth, store
from backend.app import OUTPUT_DIR, UPLOAD_DIR, app

ADMIN_EMAIL = "admin@guest-insp.local"
ADMIN_PW = "admin-password-123"
ROOT = Path(__file__).resolve().parents[1]
JPEG_BYTES = (ROOT / "hardhat_sample.jpg").read_bytes()
MP4_BYTES = (ROOT / "demo_data" / "de0000000003" / "demo-video.mp4").read_bytes()

FAKE_PHOTO_REPORT = {
    "source_image": "guest-test.jpg",
    "image_summary": "Mocked inspection for tests.",
    "people": [{
        "person_number": 1, "head_visibility": "clear",
        "helmet_observation": "visible_on_head",
        "assessment": "helmet_visible",
        "evidence": "Mocked evidence.",
    }],
    "overall_result": "all_helmets_visible",
    "human_review_required": False,
    "detection_summary": {
        "people_count": 1, "helmet_count": 1,
        "matched_count": 1, "review_count": 0,
    },
    "limitations": [],
}


def fake_inspect_image(image_path):
    return dict(FAKE_PHOTO_REPORT)


def fake_video_pipeline(video_path, inspection_id, original_filename=None):
    out = OUTPUT_DIR / inspection_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "video_inspection.json").write_text(
        '{"successful_frame_reviews": 1, "failed_frame_reviews": 0,'
        ' "overall_assessment": "Mocked video review."}',
        encoding="utf-8")
    report = out / "video_inspection_report.html"
    report.write_text("<html>mocked video report</html>", encoding="utf-8")
    return report


class FakeCap:
    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def read(self):
        return True, object()

    def release(self):
        pass


def csrf(client: TestClient) -> str:
    return client.get("/api/auth/csrf").json()["csrf_token"]


def post(client, url, payload=None):
    return client.post(url, json=payload or {},
                       headers={"x-csrf-token": csrf(client)})


def remaining(client: TestClient) -> int:
    return client.get("/api/auth/me").json()["user"]["guest_inspections"]["remaining"]


class GuestInspectionBase(unittest.TestCase):
    def setUp(self):
        Path(os.environ["SOPPROOF_DB_PATH"]).unlink(missing_ok=True)
        store.init_db()
        auth._hits.clear()
        store.create_user(ADMIN_EMAIL, auth.hash_password(ADMIN_PW), is_admin=True)
        self.admin = TestClient(app, raise_server_exceptions=False)
        assert post(self.admin, "/api/auth/login",
                    {"email": ADMIN_EMAIL, "password": ADMIN_PW}).status_code == 200
        self.created_ids = []

    def tearDown(self):
        for iid in self.created_ids:
            (UPLOAD_DIR / f"{iid}.jpg").unlink(missing_ok=True)
            (UPLOAD_DIR / f"{iid}.jpeg").unlink(missing_ok=True)
            (UPLOAD_DIR / f"{iid}.mp4").unlink(missing_ok=True)
            (OUTPUT_DIR / f"{iid}_report.html").unlink(missing_ok=True)
            shutil.rmtree(OUTPUT_DIR / iid, ignore_errors=True)

    def make_guest(self) -> TestClient:
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/guest")
        assert resp.status_code == 201, resp.text
        return client

    def register_user(self, email):
        token = post(self.admin, "/api/admin/invitations",
                     {"email": email}).json()["invite_token"]
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/register",
                    {"invite_token": token, "email": email,
                     "password": "user-password-123"})
        assert resp.status_code == 201, resp.text
        return client

    def _await_job(self, client, resp):
        """Accept-and-poll: a 202 returns the terminal status payload."""
        if resp.status_code != 202:
            return resp
        iid = resp.json()["inspection_id"]
        self.created_ids.append(iid)
        return client.get(f"/api/inspections/{iid}")

    def photo(self, client, name="site.jpg", data=JPEG_BYTES):
        with mock.patch.object(app_mod, "inspect_image", fake_inspect_image):
            resp = client.post(
                "/api/inspect/photo", files={"file": (name, data)},
                headers={"x-csrf-token": csrf(client)})
        return self._await_job(client, resp)

    def video(self, client, name="site.mp4", data=MP4_BYTES):
        with mock.patch.object(app_mod, "run_video_pipeline", fake_video_pipeline), \
                mock.patch.object(app_mod.cv2, "VideoCapture", FakeCap):
            resp = client.post(
                "/api/inspect/video", files={"file": (name, data)},
                headers={"x-csrf-token": csrf(client)})
        return self._await_job(client, resp)


class GuestCreditTests(GuestInspectionBase):
    def test_1_new_guest_receives_exactly_two_credits(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/guest")
        self.assertEqual(resp.status_code, 201)
        allowance = resp.json()["user"]["guest_inspections"]
        self.assertEqual(allowance, {"limit": 2, "used": 0, "remaining": 2})
        self.assertEqual(remaining(client), 2)

    def test_2_guest_can_run_real_photo_inspection(self):
        guest = self.make_guest()
        resp = self.photo(guest)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["guest_inspections"]["remaining"], 1)
        self.assertEqual(remaining(guest), 1)

    def test_3_guest_can_run_real_video_inspection(self):
        guest = self.make_guest()
        resp = self.video(guest)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["guest_inspections"]["remaining"], 1)

    def test_4_photo_and_video_share_one_allowance(self):
        guest = self.make_guest()
        self.assertEqual(self.photo(guest).status_code, 200)
        self.assertEqual(self.video(guest).status_code, 200)
        self.assertEqual(remaining(guest), 0)
        self.assertEqual(self.photo(guest).status_code, 403)

    def test_5_third_inspection_rejected_403(self):
        guest = self.make_guest()
        self.assertEqual(self.photo(guest).status_code, 200)
        self.assertEqual(self.photo(guest).status_code, 200)
        third = self.photo(guest)
        self.assertEqual(third.status_code, 403)
        self.assertIn("limit reached", third.json()["detail"].lower())

    def test_6_invalid_upload_does_not_consume_credit(self):
        guest = self.make_guest()
        bad_type = guest.post(
            "/api/inspect/photo", files={"file": ("evil.exe", b"x")},
            headers={"x-csrf-token": csrf(guest)})
        self.assertEqual(bad_type.status_code, 400)
        bad_image = guest.post(
            "/api/inspect/photo", files={"file": ("site.jpg", b"\xff\xd8\xff")},
            headers={"x-csrf-token": csrf(guest)})
        self.assertEqual(bad_image.status_code, 400)
        bad_video = guest.post(
            "/api/inspect/video", files={"file": ("clip.mp4", b"not a video")},
            headers={"x-csrf-token": csrf(guest)})
        self.assertEqual(bad_video.status_code, 400)
        self.assertEqual(remaining(guest), 2)

    def test_7_refresh_and_new_tab_do_not_reset_credits(self):
        guest = self.make_guest()
        self.assertEqual(self.photo(guest).status_code, 200)
        # Simulate refresh / second tab: new client, same session cookie.
        raw = guest.cookies.get(auth.SESSION_COOKIE)
        tab = TestClient(app, raise_server_exceptions=False)
        tab.cookies.set(auth.SESSION_COOKIE, raw)
        self.assertEqual(remaining(tab), 1)
        self.assertEqual(self.photo(tab).status_code, 200)
        self.assertEqual(self.photo(tab).status_code, 403)

    def test_8_simultaneous_requests_cannot_exceed_two(self):
        # Store-level atomicity: 4 racing consumers, exactly 2 succeed.
        token = auth.token_hash("race-session")
        store.create_guest_session(token, time.time() + 3600)
        results, barrier = [], threading.Barrier(4)

        def consume():
            barrier.wait()
            results.append(store.consume_guest_credit(token))

        threads = [threading.Thread(target=consume) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), [False, False, True, True])
        self.assertEqual(store.guest_credits_used(token), 2)
        # HTTP-level: racing photo posts from one guest session.
        guest = self.make_guest()
        raw = guest.cookies.get(auth.SESSION_COOKIE)
        statuses = []

        def hit():
            c = TestClient(app, raise_server_exceptions=False)
            c.cookies.set(auth.SESSION_COOKIE, raw)
            with mock.patch.object(app_mod, "inspect_image", fake_inspect_image):
                resp = c.post(
                    "/api/inspect/photo",
                    files={"file": ("site.jpg", JPEG_BYTES)},
                    headers={"x-csrf-token": csrf(c)})
            if resp.status_code == 202:
                self.created_ids.append(resp.json()["inspection_id"])
            statuses.append(resp.status_code)

        racers = [threading.Thread(target=hit) for _ in range(3)]
        for t in racers:
            t.start()
        for t in racers:
            t.join()
        self.assertEqual(sorted(statuses), [202, 202, 403])


class GuestOwnershipTests(GuestInspectionBase):
    def test_9_guest_b_cannot_access_guest_a_reports(self):
        guest_a, guest_b = self.make_guest(), self.make_guest()
        resp = self.photo(guest_a)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(guest_b.get(data["report_url"]).status_code, 404)
        self.assertEqual(guest_b.get(data["image_url"]).status_code, 404)
        b_urls = [r["url"] for r in guest_b.get("/api/reports").json()["reports"]]
        self.assertFalse(any(data["inspection_id"] in u for u in b_urls))

    def test_10_guest_cannot_access_authenticated_user_reports(self):
        user = self.register_user("owner2@guest-insp.local")
        resp = self.photo(user)
        self.assertEqual(resp.status_code, 200)
        guest = self.make_guest()
        self.assertEqual(guest.get(resp.json()["report_url"]).status_code, 404)
        self.assertEqual(guest.get(resp.json()["image_url"]).status_code, 404)

    def test_11_guest_can_access_own_reports_and_evidence(self):
        guest = self.make_guest()
        resp = self.photo(guest)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(guest.get(data["report_url"]).status_code, 200)
        self.assertEqual(guest.get(data["image_url"]).status_code, 200)
        urls = [r["url"] for r in guest.get("/api/reports").json()["reports"]]
        self.assertTrue(any(data["inspection_id"] in u for u in urls))

    def test_12_expired_guest_session_loses_access(self):
        guest = self.make_guest()
        resp = self.photo(guest)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        token = auth.token_hash(guest.cookies.get(auth.SESSION_COOKIE))
        with store.connect() as conn:
            conn.execute(
                "UPDATE guest_sessions SET expires_at = ? WHERE token_hash = ?",
                (time.time() - 10, token))
        self.assertEqual(guest.get("/api/auth/me").status_code, 401)
        self.assertEqual(guest.get(data["report_url"]).status_code, 401)
        self.assertEqual(guest.get(data["image_url"]).status_code, 401)
        again = guest.post(
            "/api/inspect/photo", files={"file": ("site.jpg", JPEG_BYTES)},
            headers={"x-csrf-token": csrf(guest)})
        self.assertEqual(again.status_code, 401)


class JobPollingTests(GuestInspectionBase):
    def test_16_job_polling_is_owner_only(self):
        guest_a, guest_b = self.make_guest(), self.make_guest()
        resp = self.photo(guest_a)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "completed")
        iid = resp.json()["inspection_id"]
        self.assertEqual(guest_a.get(f"/api/inspections/{iid}").status_code, 200)
        self.assertEqual(guest_b.get(f"/api/inspections/{iid}").status_code, 404)
        user = self.register_user("poller@guest-insp.local")
        self.assertEqual(user.get(f"/api/inspections/{iid}").status_code, 404)
        anon = TestClient(app, raise_server_exceptions=False)
        self.assertEqual(anon.get(f"/api/inspections/{iid}").status_code, 401)

    def test_17_failed_job_status_and_credit_stays_consumed(self):
        guest = self.make_guest()

        def boom(image_path):
            raise RuntimeError("secret internal pipeline detail")

        with mock.patch.object(app_mod, "inspect_image", boom):
            resp = guest.post(
                "/api/inspect/photo",
                files={"file": ("site.jpg", JPEG_BYTES)},
                headers={"x-csrf-token": csrf(guest)})
        self.assertEqual(resp.status_code, 202)
        iid = resp.json()["inspection_id"]
        self.created_ids.append(iid)
        status = guest.get(f"/api/inspections/{iid}")
        self.assertEqual(status.status_code, 200)
        body = status.json()
        self.assertEqual(body["status"], "failed")
        self.assertNotIn("secret internal", body.get("error", ""))
        # Accepted-then-failed still consumed the credit (documented).
        self.assertEqual(remaining(guest), 1)
        # Failure cleanup removed the uploaded artifact.
        self.assertFalse((UPLOAD_DIR / f"{iid}.jpg").exists())


class UnchangedBehaviourTests(GuestInspectionBase):
    def test_13_authenticated_user_inspection_unchanged(self):
        user = self.register_user("plain@guest-insp.local")
        resp = self.photo(user)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("guest_inspections", resp.json())
        self.assertEqual(user.get(resp.json()["report_url"]).status_code, 200)
        me = user.get("/api/auth/me").json()["user"]
        self.assertNotIn("guest_inspections", me)

    def test_14_admin_functionality_unchanged(self):
        self.assertEqual(self.admin.get("/api/auth/me").json()["user"]["email"],
                         ADMIN_EMAIL)
        self.assertEqual(self.admin.get("/api/admin/invitations").status_code, 200)
        guest = self.make_guest()
        self.assertEqual(guest.get("/api/admin/invitations").status_code, 404)
        admin_urls = [r["url"] for r in self.admin.get("/api/reports").json()["reports"]]
        self.assertFalse(any("de00000000" in u for u in admin_urls))

    def test_15_synthetic_demo_replay_does_not_consume_credit(self):
        guest = self.make_guest()
        self.assertEqual(
            guest.get("/outputs/de0000000001_report.html").status_code, 200)
        self.assertEqual(guest.get("/uploads/de0000000009.jpg").status_code, 200)
        self.assertEqual(
            guest.get("/outputs/de0000000003/outputs/video_review_frames/frame_0000.jpg").status_code, 200)
        self.assertEqual(remaining(guest), 2)


if __name__ == "__main__":
    unittest.main()
