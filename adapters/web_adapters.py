"""
Web / HTTP Adapters — Real HTTP requests using only stdlib urllib.
"""

import json
import urllib.request
import urllib.error
import urllib.parse
import ssl
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config


def _timeout():
    return config.HTTP_TIMEOUT


def _make_ssl_ctx():
    ctx = ssl.create_default_context()
    return ctx


# ── adapters ───────────────────────────────────────────────────────────────────

def http_get_adapter(step: dict):
    """
    Perform an HTTP GET request.
    args: {url: str, headers: dict (optional), params: dict (optional)}
    Returns the response body as a string (JSON auto-parsed if content-type matches).
    """
    args    = step.get("args", {})
    url     = str(args.get("url", ""))
    headers = args.get("headers") or {}
    params  = args.get("params") or {}

    if not url:
        raise ValueError("http_get: 'url' is required.")

    if params:
        encoded = urllib.parse.urlencode(params)
        url = f"{url}?{encoded}" if "?" not in url else f"{url}&{encoded}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_timeout(), context=_make_ssl_ctx()) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            print(f"[Web] GET {url} → {resp.status}")
            if "application/json" in content_type:
                try:
                    return json.loads(body)
                except Exception:
                    pass
            return body
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='ignore')}")


def http_post_adapter(step: dict):
    """
    Perform an HTTP POST request.
    args: {url: str, body: dict|str, headers: dict (optional), json_body: bool (default True)}
    """
    args     = step.get("args", {})
    url      = str(args.get("url", ""))
    body     = args.get("body") or args.get("data") or {}
    headers  = dict(args.get("headers") or {})
    as_json  = args.get("json_body", True)

    if not url:
        raise ValueError("http_post: 'url' is required.")

    if as_json:
        payload = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    else:
        payload = urllib.parse.urlencode(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_timeout(), context=_make_ssl_ctx()) as resp:
            body_resp = resp.read().decode("utf-8", errors="replace")
            print(f"[Web] POST {url} → {resp.status}")
            try:
                return json.loads(body_resp)
            except Exception:
                return body_resp
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='ignore')}")


def download_file_adapter(step: dict):
    """
    Download a file from a URL to a local path.
    args: {url: str, path: str}
    """
    args = step.get("args", {})
    url  = str(args.get("url", ""))
    path = str(args.get("path") or args.get("dest") or "")
    if not url or not path:
        raise ValueError("download_file: 'url' and 'path' are required.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    print(f"[Web] Downloaded {url} → {path} ({size} bytes)")
    return f"Downloaded to {path} ({size} bytes)"


# ── registration ───────────────────────────────────────────────────────────────

def setup_web_adapters(toolgate):
    toolgate.register_adapter("http_get",       http_get_adapter)
    toolgate.register_adapter("http_post",      http_post_adapter)
    toolgate.register_adapter("web_request",    http_get_adapter)  # alias
    toolgate.register_adapter("download_file",  download_file_adapter)
    print("[Adapters] Web/HTTP adapters registered.")
