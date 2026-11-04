"""One-time bootstrap: create the first SOPProof administrator account.

Usage:
    .venv/bin/python -m backend.bootstrap_admin --email admin@example.com

The password is read interactively with getpass (never from argv, never
hardcoded, never printed). Refuses to run once any user exists.
"""
import argparse
import getpass
import sys

from backend import auth, store


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the first SOPProof admin.")
    parser.add_argument("--email", required=True, help="Administrator email address.")
    args = parser.parse_args()

    store.init_db()
    if store.count_users() > 0:
        print("Refusing: user accounts already exist. Bootstrap is one-time only.")
        return 1

    email = args.email.strip().lower()
    password = getpass.getpass("New admin password (min 12 characters): ")
    if len(password) < 12:
        print("Password must be at least 12 characters.")
        return 1
    if password != getpass.getpass("Confirm admin password: "):
        print("Passwords do not match.")
        return 1

    user_id = store.create_user(email, auth.hash_password(password), is_admin=True)
    print(f"Administrator created (id={user_id}, email={email}).")
    print("You can now sign in and create registration invitations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
