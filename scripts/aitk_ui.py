"""Minimal client for the AI Toolkit UI's HTTP API, shared by the import_* scripts.

The UI (Next.js) owns aitk_db.db, so the scripts go through its API rather
than writing the database themselves. Environment:
  AITK_UI_URL       base URL of the UI (default: the first of
                    http://127.0.0.1:3000 (dev/systemd) and :8675 (docker image,
                    `npm run start`) that answers)
  AI_TOOLKIT_AUTH   bearer token, if the UI is password-protected; falls back
                    to the value in ~/.config/ai-toolkit.env (the file the
                    systemd units load), then to the container's environment
                    (/proc/1/environ), where docker/RunPod put it and where an
                    ssh session does not see it
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


DEFAULT_URLS = ("http://127.0.0.1:3000", "http://127.0.0.1:8675")
_base_url = None


def _answers(url):
    try:
        urllib.request.urlopen(url + "/api/settings", timeout=5).close()
        return True
    except urllib.error.HTTPError:
        return True  # it is up, just wants a token
    except (urllib.error.URLError, OSError):
        return False


def base_url():
    global _base_url
    if _base_url is None:
        env = os.environ.get("AITK_UI_URL")
        if env:
            _base_url = env.rstrip("/")
        else:
            _base_url = next((u for u in DEFAULT_URLS if _answers(u)), DEFAULT_URLS[0])
    return _base_url


def _read_container_env(key):
    try:
        with open("/proc/1/environ", "rb") as f:
            for item in f.read().split(b"\0"):
                k, _, v = item.decode(errors="replace").partition("=")
                if k == key:
                    return v
    except OSError:
        pass
    return None


def _auth_token():
    return (os.environ.get("AI_TOOLKIT_AUTH") or _read_env_file("AI_TOOLKIT_AUTH")
            or _read_container_env("AI_TOOLKIT_AUTH"))


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


def get_job(job_id):
    return request("GET", f"/api/jobs?id={job_id}")


def start_job(job_id, gpu_ids):
    """Same as the UI's Start button: queue the job, make sure its GPU queue runs."""
    request("GET", f"/api/jobs/{job_id}/start")
    request("GET", f"/api/queue/{gpu_ids}/start")


def stop_job(job_id):
    return request("GET", f"/api/jobs/{job_id}/stop")


def gpu_count():
    """Number of GPUs the UI sees (nvidia-smi or rocm-smi), or None."""
    try:
        return len(request("GET", "/api/gpu").get("gpus") or []) or None
    except UIError:
        return None


def save_job(**fields):
    """POST /api/jobs: creates a job, or updates it in place when `id` is given."""
    return request("POST", "/api/jobs", fields)
