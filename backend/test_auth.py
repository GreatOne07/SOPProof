"""Authentication and multi-user privacy tests. No network/AI calls.

Run: .venv/bin/python -m unittest backend.test_auth -v
"""
import glob
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ["SOPPROOF_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test-auth.db")

from fastapi.testclient import TestClient

from backend import auth, store
from backend.app import OUTPUT_DIR, UPLOAD_DIR, app

ADMIN_EMAIL = "admin@test.local"
ADMIN_PW = "admin-password-123"


def csrf(client: TestClient) -> str:
    resp = client.get("/api/auth/csrf")
    assert resp.status_code == 200
    return resp.json()["csrf_token"]


def post(client, url, payload=None):
    return client.post(url, json=payload or {}, headers={"x-csrf-token": csrf(client)})


def make_user(client, email, password, invite_token):
    resp = post(client, "/api/auth/register",
                {"invite_token": invite_token, "email": email, "password": password})
    assert resp.status_code == 201, resp.text
    return resp


class AuthTestBase(unittest.TestCase):
    def setUp(self):
        db = Path(os.environ["SOPPROOF_DB_PATH"])
        db.unlink(missing_ok=True)
        store.init_db()
        auth._hits.clear()
        self.admin_id = store.create_user(
            ADMIN_EMAIL, auth.hash_password(ADMIN_PW), is_admin=True
        )
        self.admin = TestClient(app, raise_server_exceptions=False)
        resp = post(self.admin, "/api/auth/login",
                    {"email": ADMIN_EMAIL, "password": ADMIN_PW})
        assert resp.status_code == 200, resp.text

    def invite(self, email=None):
        resp = post(self.admin, "/api/admin/invitations", {"email": email})
        assert resp.status_code == 201, resp.text
        return resp.json()

    def register_user(self, email, password="user-password-123"):
        token = self.invite(email)["invite_token"]
        client = TestClient(app, raise_server_exceptions=False)
        make_user(client, email, password, token)
        return client


class UnauthenticatedTests(AuthTestBase):
    def test_private_endpoints_reject_anonymous(self):
        anon = TestClient(app, raise_server_exceptions=False)
        self.assertEqual(anon.get("/api/reports").status_code, 401)
        self.assertEqual(anon.get("/api/auth/me").status_code, 401)
        self.assertEqual(
            anon.post("/api/inspect/photo",
                      files={"file": ("x.jpg", b"x")}).status_code, 401)
        self.assertEqual(
            anon.post("/api/inspect/video",
                      files={"file": ("x.mp4", b"x")}).status_code, 401)
        self.assertEqual(anon.get("/uploads/00fa05d34209.jpg").status_code, 401)
        self.assertEqual(
            anon.get("/outputs/00fa05d34209_report.html").status_code, 401)
        self.assertEqual(anon.get("/api/health").status_code, 200)

    def test_internal_outputs_still_blocked(self):
        for path in ("/outputs/video_inspection.json",
                     "/outputs/954010f3b9fc/outputs/video_inspection_report.html.bak"):
            self.assertEqual(self.admin.get(path).status_code, 404, path)


class InviteTests(AuthTestBase):
    def test_register_requires_valid_invite(self):
        anon = TestClient(app, raise_server_exceptions=False)
        resp = post(anon, "/api/auth/register",
                    {"invite_token": "nope", "email": "x@test.local",
                     "password": "user-password-123"})
        self.assertEqual(resp.status_code, 400)

    def test_invite_single_use_and_revocation(self):
        token = self.invite("a@test.local")["invite_token"]
        client = TestClient(app, raise_server_exceptions=False)
        make_user(client, "a@test.local", "user-password-123", token)
        again = TestClient(app, raise_server_exceptions=False)
        resp = post(again, "/api/auth/register",
                    {"invite_token": token, "email": "b@test.local",
                     "password": "user-password-123"})
        self.assertEqual(resp.status_code, 400)

        inv2 = self.invite()
        resp = self.admin.delete(
            f"/api/admin/invitations/{inv2['id']}",
            headers={"x-csrf-token": csrf(self.admin)})
        self.assertEqual(resp.status_code, 200)
        third = TestClient(app, raise_server_exceptions=False)
        resp = post(third, "/api/auth/register",
                    {"invite_token": inv2["invite_token"], "email": "c@test.local",
                     "password": "user-password-123"})
        self.assertEqual(resp.status_code, 400)

    def test_invite_tokens_never_in_list_or_logs(self):
        inv = self.invite("list@test.local")
        listing = self.admin.get("/api/admin/invitations").json()
        self.assertNotIn(inv["invite_token"], str(listing))
        for item in listing["invitations"]:
            self.assertNotIn("token", item)

    def test_non_admin_cannot_manage_invites(self):
        user = self.register_user("plain@test.local")
        self.assertEqual(post(user, "/api/admin/invitations", {}).status_code, 404)
        self.assertEqual(user.get("/api/admin/invitations").status_code, 404)


class SessionTests(AuthTestBase):
    def test_password_stored_as_argon2id_only(self):
        row = store.get_user_by_email(ADMIN_EMAIL)
        self.assertTrue(row["password_hash"].startswith("$argon2id$"))
        self.assertNotIn(ADMIN_PW, row["password_hash"])

    def test_session_cookie_is_httponly_samesite(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/login",
                    {"email": ADMIN_EMAIL, "password": ADMIN_PW})
        cookies = "; ".join(resp.headers.get_list("set-cookie"))
        self.assertIn("sopproof_session=", cookies)
        self.assertIn("HttpOnly", cookies)
        self.assertIn("SameSite=lax", cookies)

    def test_csrf_required_for_state_changes(self):
        client = self.register_user("csrf@test.local")
        no_csrf = client.post("/api/inspect/photo", files={"file": ("evil.exe", b"x")})
        self.assertEqual(no_csrf.status_code, 403)
        with_csrf = client.post(
            "/api/inspect/photo", files={"file": ("evil.exe", b"x")},
            headers={"x-csrf-token": csrf(client)})
        self.assertEqual(with_csrf.status_code, 400)  # validation preserved
        self.assertEqual(client.post("/api/auth/logout").status_code, 403)
        self.assertEqual(post(client, "/api/auth/logout").status_code, 200)
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_expired_session_rejected(self):
        user = store.get_user_by_email(ADMIN_EMAIL)
        store.create_session(auth.token_hash("expired-token"), user["id"],
                             time.time() - 10)
        client = TestClient(app, raise_server_exceptions=False)
        client.cookies.set("sopproof_session", "expired-token")
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_wrong_password_and_rate_limit(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = post(client, "/api/auth/login",
                    {"email": ADMIN_EMAIL, "password": "wrong-password-xx"})
        self.assertEqual(resp.status_code, 401)
        status = None
        for _ in range(12):
            status = post(client, "/api/auth/login",
                          {"email": "rl@test.local",
                           "password": "wrong-password-xx"}).status_code
        self.assertEqual(status, 429)


class OwnershipTests(AuthTestBase):
    FAKE_ID = "abcdef123456"

    def tearDown(self):
        (OUTPUT_DIR / f"{self.FAKE_ID}_report.html").unlink(missing_ok=True)
        (UPLOAD_DIR / f"{self.FAKE_ID}.jpg").unlink(missing_ok=True)

    def test_two_user_isolation_with_synthetic_report(self):
        user_a = self.register_user("a@test.local")
        user_b = self.register_user("b@test.local")
        a_id = store.get_user_by_email("a@test.local")["id"]
        (OUTPUT_DIR / f"{self.FAKE_ID}_report.html").write_text("<html>A</html>")
        (UPLOAD_DIR / f"{self.FAKE_ID}.jpg").write_bytes(b"\xff\xd8fake")
        store.record_inspection(self.FAKE_ID, a_id, "photo")

        self.assertEqual(
            user_a.get(f"/outputs/{self.FAKE_ID}_report.html").status_code, 200)
        self.assertEqual(
            user_b.get(f"/outputs/{self.FAKE_ID}_report.html").status_code, 404)
        self.assertEqual(user_a.get(f"/uploads/{self.FAKE_ID}.jpg").status_code, 200)
        self.assertEqual(user_b.get(f"/uploads/{self.FAKE_ID}.jpg").status_code, 404)
        a_ids = [r["url"] for r in user_a.get("/api/reports").json()["reports"]]
        b_ids = [r["url"] for r in user_b.get("/api/reports").json()["reports"]]
        self.assertTrue(any(self.FAKE_ID in u for u in a_ids))
        self.assertFalse(any(self.FAKE_ID in u for u in b_ids))

    def test_legacy_records_hidden_until_claimed_then_admin_only(self):
        from backend.claim_legacy import main as claim_legacy

        user_a = self.register_user("legacy@test.local")
        legacy = sorted(glob.glob(str(OUTPUT_DIR / "*_report.html")))
        if not legacy:
            self.skipTest("no legacy photo reports on disk")
        name = Path(legacy[0]).name
        self.assertEqual(user_a.get(f"/outputs/{name}").status_code, 404)
        self.assertEqual(user_a.get("/api/reports").json()["total"], 0)

        self.assertEqual(claim_legacy(), 0)
        self.assertEqual(self.admin.get(f"/outputs/{name}").status_code, 200)
        self.assertEqual(user_a.get(f"/outputs/{name}").status_code, 404)
        self.assertGreater(self.admin.get("/api/reports").json()["total"], 0)
        self.assertEqual(claim_legacy(), 0)  # rerun is a no-op


if __name__ == "__main__":
    unittest.main()
