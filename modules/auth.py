"""
auth.py
--------
Registration / login / logout using bcrypt password hashing.
Never stores plain-text passwords. Never returns password_hash to callers.
"""

from __future__ import annotations

import re
import bcrypt

from modules import database


USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,20}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any expected auth failure (bad password, duplicate user, etc.)."""


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def register_user(username: str, email: str, password: str, confirm_password: str) -> int:
    """Validates input, hashes the password, and creates the user. Returns new user id."""
    username = username.strip()
    email = email.strip().lower()

    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3-20 characters: letters, numbers, underscore only.")
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters long.")
    if password != confirm_password:
        raise AuthError("Passwords do not match.")
    if database.get_user_by_username(username):
        raise AuthError("That username is already taken.")
    if database.get_user_by_email(email):
        raise AuthError("That email is already registered.")

    password_hash = _hash_password(password)
    return database.create_user(username, email, password_hash)


def login_user(username: str, password: str) -> dict:
    """Returns a dict with safe user fields (no password hash) on success."""
    username = username.strip()
    row = database.get_user_by_username(username)
    if row is None:
        raise AuthError("Invalid username or password.")
    if not _verify_password(password, row["password_hash"]):
        raise AuthError("Invalid username or password.")

    return {"id": row["id"], "username": row["username"], "email": row["email"]}


def logout_user(session_state) -> None:
    """Clears auth-related keys from Streamlit's session_state."""
    for key in ("user", "logged_in"):
        if key in session_state:
            del session_state[key]
