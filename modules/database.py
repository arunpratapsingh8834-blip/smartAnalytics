"""
database.py
------------
Thin SQLite wrapper for the app. Kept deliberately simple (raw SQL, no ORM)
so it's easy to explain in a viva, but all queries go through this one
module so SQLite could later be swapped for PostgreSQL without touching
the rest of the codebase.
"""

from __future__ import annotations

import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "app.db")


@contextmanager
def get_connection():
    """Yields a sqlite3 connection with foreign keys enabled, always closed after use."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates all tables if they do not already exist. Safe to call every app start."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cleaning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                original_problem TEXT,
                recommended_method TEXT,
                selected_method TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS forecast_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                target TEXT NOT NULL,
                model TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                metrics TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
            );
            """
        )


# ---------------------------------------------------------------- users ----

def create_user(username: str, email: str, password_hash: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()


# ------------------------------------------------------------- datasets ----

def create_dataset(user_id: int, filename: str, rows: int, columns: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO datasets (user_id, filename, upload_time, rows, columns) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, datetime.utcnow().isoformat(), rows, columns),
        )
        return cur.lastrowid


def list_datasets(user_id: int):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM datasets WHERE user_id = ? ORDER BY upload_time DESC",
            (user_id,),
        ).fetchall()


# ------------------------------------------------------ cleaning_history ---

def log_cleaning_action(
    dataset_id: int,
    column_name: str,
    original_problem: str,
    recommended_method: str,
    selected_method: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO cleaning_history "
            "(dataset_id, column_name, original_problem, recommended_method, selected_method, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                column_name,
                original_problem,
                recommended_method,
                selected_method,
                datetime.utcnow().isoformat(),
            ),
        )


def get_cleaning_history(dataset_id: int):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM cleaning_history WHERE dataset_id = ? ORDER BY created_at",
            (dataset_id,),
        ).fetchall()


# ------------------------------------------------------ forecast_history ---

def log_forecast_run(
    dataset_id: int, target: str, model: str, horizon: int, metrics: dict[str, Any]
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO forecast_history (dataset_id, target, model, horizon, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                target,
                model,
                horizon,
                json.dumps(metrics),
                datetime.utcnow().isoformat(),
            ),
        )


def get_forecast_history(dataset_id: int):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM forecast_history WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,),
        ).fetchall()
