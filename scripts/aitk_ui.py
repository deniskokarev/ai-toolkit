"""Minimal client for the AI Toolkit UI's HTTP API, shared by the import_* scripts.

The UI (Next.js, default port 3000) owns aitk_db.db, so the scripts go through
its API rather than writing the database themselves. Environment:
  AITK_UI_URL       base URL of the UI (default http://127.0.0.1:3000)
  AI_TOOLKIT_AUTH   bearer token, if the UI is password-protected; falls back
                    to the value in ~/.config/ai-toolkit.env (the file the
                    systemd units load)
"""
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(TOOLKIT_ROOT, "aitk_db.db")
ENV_FILE = os.path.expanduser("~/.config/ai-toolkit.env")


class UIError(Exception):
    pass


def _read_env_file(key):
    try:
        with open(ENV_FILE) as f:
            for line in f:
                m = re.match(rf"^\s*(?:export\s+)?{key}\s*=\s*(.*?)\s*$", line)
                if m:
                    return m.group(1).strip("'\"")
    except OSError:
        pass
    return None


def base_url():
    return os.environ.get("AITK_UI_URL", "http://127.0.0.1:3000").rstrip("/")


def _auth_token():
    return os.environ.get("AI_TOOLKIT_AUTH") or _read_env_file("AI_TOOLKIT_AUTH")


def request(method, path, body=None):
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    token = _auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", "")
        except Exception:
            msg = ""
        raise UIError(f"{method} {path} -> HTTP {e.code} {msg}".strip()) from None
    except urllib.error.URLError as e:
        raise UIError(
            f"cannot reach the UI at {base_url()} ({e.reason}); "
            "is ai-toolkit-ui.service running? (set AITK_UI_URL to override)"
        ) from None


def get_settings():
    return request("GET", "/api/settings")


def get_settings_offline():
    """Settings straight from the DB, for when the UI is down (read-only)."""
    out = {}
    if os.path.exists(DB_PATH):
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            out = {k: v for k, v in con.execute("SELECT key, value FROM Settings") if v}
        finally:
            con.close()
    out.setdefault("DATASETS_FOLDER", os.path.join(TOOLKIT_ROOT, "datasets"))
    out.setdefault("TRAINING_FOLDER", os.path.join(TOOLKIT_ROOT, "output"))
    return out


def list_jobs():
    return request("GET", "/api/jobs")["jobs"]


def find_job(name):
    return next((j for j in list_jobs() if j["name"] == name), None)


def save_job(**fields):
    """POST /api/jobs: creates a job, or updates it in place when `id` is given."""
    return request("POST", "/api/jobs", fields)
