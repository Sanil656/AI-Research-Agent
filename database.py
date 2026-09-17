"""
MySQL Database & Authentication Manager for the Research AI Agent.

Handles:
1. MySQL connection pooling and schema initialization (users, user_threads).
2. Industry-standard salted bcrypt password hashing and verification.
3. User registration, authentication, and user-isolated thread persistence.
"""

import os
import json
import uuid
from typing import Optional, Dict, Any, List, Tuple
import bcrypt
from dotenv import load_dotenv

load_dotenv()


def _get_config_val(key: str, default: str = "") -> str:
    """Retrieves configuration from environment variables or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        import streamlit as st
        # Check top-level or nested [mysql] section
        if hasattr(st, "secrets"):
            if key in st.secrets:
                return str(st.secrets[key]).strip()
            if "mysql" in st.secrets and key in st.secrets["mysql"]:
                return str(st.secrets["mysql"][key]).strip()
            # Map standard env names to secrets.toml fields
            key_map = {
                "MYSQL_HOST": "host",
                "MYSQL_PORT": "port",
                "MYSQL_USER": "user",
                "MYSQL_PASSWORD": "password",
                "MYSQL_DATABASE": "database"
            }
            if key in key_map and "mysql" in st.secrets and key_map[key] in st.secrets["mysql"]:
                return str(st.secrets["mysql"][key_map[key]]).strip()
    except Exception:
        pass
    return default


def get_db_connection():
    """
    Establishes a MySQL/TiDB database connection using configured credentials.
    Returns the connection object or raises an exception.
    """
    import mysql.connector

    host = _get_config_val("MYSQL_HOST", "localhost")
    port = int(_get_config_val("MYSQL_PORT", "4000" if "tidb" in host.lower() else "3306"))
    user = _get_config_val("MYSQL_USER", "root")
    password = _get_config_val("MYSQL_PASSWORD", "")
    database = _get_config_val("MYSQL_DATABASE", "test" if "tidb" in host.lower() else "research_agent_db")

    conn_kwargs = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "autocommit": True
    }
    
    # Enable SSL for remote cloud databases (TiDB / Aiven / AWS)
    if host not in ("localhost", "127.0.0.1"):
        conn_kwargs["ssl_disabled"] = False

    return mysql.connector.connect(**conn_kwargs)


def init_db() -> bool:
    """
    Initializes the database and required tables if they do not already exist.
    Creates `users` and `user_threads` tables.
    """
    try:
        import mysql.connector

        # 1. Try connecting directly to target database
        try:
            conn = get_db_connection()
        except Exception:
            # If database doesn't exist on local MySQL, try creating it
            host = _get_config_val("MYSQL_HOST", "localhost")
            port = int(_get_config_val("MYSQL_PORT", "3306"))
            user = _get_config_val("MYSQL_USER", "root")
            password = _get_config_val("MYSQL_PASSWORD", "")
            database = _get_config_val("MYSQL_DATABASE", "research_agent_db")

            admin_conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                autocommit=True
            )
            cur = admin_conn.cursor()
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4;")
            cur.close()
            admin_conn.close()
            conn = get_db_connection()
        cur = conn.cursor()

        # 1. Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `users` (
                `id` VARCHAR(36) PRIMARY KEY,
                `username` VARCHAR(50) NOT NULL UNIQUE,
                `email` VARCHAR(100) NOT NULL UNIQUE,
                `password_hash` VARCHAR(255) NOT NULL,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `last_login` DATETIME NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        # 2. User Research Threads Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `user_threads` (
                `thread_id` VARCHAR(36) PRIMARY KEY,
                `user_id` VARCHAR(36) NOT NULL,
                `title` VARCHAR(255) NOT NULL,
                `messages_json` LONGTEXT NOT NULL,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_id (`user_id`),
                CONSTRAINT fk_user FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)

        cur.close()
        conn.close()
        return True
    except Exception as err:
        print(f"[Database Init Notice] MySQL connection not active: {err}")
        return False


def register_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Registers a new user with salted bcrypt password hashing in MySQL.
    
    Returns:
        (success: bool, message: str)
    """
    username = username.strip()
    email = email.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        # Check for existing username or email
        cur.execute("SELECT id FROM users WHERE username = %s OR email = %s LIMIT 1", (username, email))
        if cur.fetchone():
            cur.close()
            conn.close()
            return False, "Username or email is already registered."

        # Generate salted bcrypt hash
        pw_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt(rounds=12)
        hashed_pw = bcrypt.hashpw(pw_bytes, salt).decode("utf-8")

        user_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (%s, %s, %s, %s)",
            (user_id, username, email, hashed_pw)
        )
        cur.close()
        conn.close()
        return True, "Account registered successfully! Please log in."
    except Exception as err:
        return False, f"Database error: {str(err)}"


def authenticate_user(username_or_email: str, password: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Authenticates a user against MySQL credentials using bcrypt.
    
    Returns:
        (user_dict or None, message: str)
    """
    identifier = username_or_email.strip()
    if not identifier or not password:
        return None, "Please fill in all fields."

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE username = %s OR email = %s LIMIT 1",
            (identifier, identifier.lower())
        )
        user = cur.fetchone()

        if not user:
            cur.close()
            conn.close()
            return None, "Invalid username/email or password."

        stored_hash = user["password_hash"]
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
            # Update last login timestamp
            cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
            cur.close()
            conn.close()

            user_data = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "created_at": str(user["created_at"])
            }
            return user_data, "Login successful!"

        cur.close()
        conn.close()
        return None, "Invalid username/email or password."
    except Exception as err:
        return None, f"Database error: {str(err)}"


def save_user_thread(user_id: str, thread_id: str, title: str, messages: List[Dict[str, Any]]) -> bool:
    """Saves or updates a research conversation thread in MySQL for a specific user."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        messages_json = json.dumps(messages)

        cur.execute("""
            INSERT INTO user_threads (thread_id, user_id, title, messages_json)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE title = VALUES(title), messages_json = VALUES(messages_json), updated_at = NOW();
        """, (thread_id, user_id, title, messages_json))

        cur.close()
        conn.close()
        return True
    except Exception as err:
        print(f"[Thread Save Warning] {err}")
        return False


def load_user_threads(user_id: str) -> Dict[str, Dict[str, Any]]:
    """Loads all saved research conversation threads for a specific user from MySQL."""
    threads = {}
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute(
            "SELECT thread_id, title, messages_json, updated_at FROM user_threads WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = cur.fetchall()

        for r in rows:
            try:
                msgs = json.loads(r["messages_json"])
            except Exception:
                msgs = []
            threads[r["thread_id"]] = {
                "id": r["thread_id"],
                "title": r["title"],
                "created_at": str(r["updated_at"])[:16],
                "messages": msgs
            }

        cur.close()
        conn.close()
    except Exception as err:
        print(f"[Thread Load Notice] {err}")
    return threads


def delete_user_thread(user_id: str, thread_id: str) -> bool:
    """Deletes a research conversation thread belonging to a user."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_threads WHERE thread_id = %s AND user_id = %s", (thread_id, user_id))
        cur.close()
        conn.close()
        return True
    except Exception as err:
        print(f"[Thread Delete Warning] {err}")
        return False
