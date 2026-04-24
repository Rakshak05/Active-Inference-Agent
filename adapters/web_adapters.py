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
            
            if "text/html" in content_type:
                import re
                # Naive text extraction
                text = re.sub(r'<(script|style|nav|footer|header|aside|iframe).*?</\1>', '', body, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                # Return only a reasonable chunk for processing
                return {"content": text[:50000], "status": 200, "url": url}
            
            return {"content": body, "status": 200, "url": url}

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


def web_search_adapter(step: dict):
    """
    Perform a public web search via DuckDuckGo.
    args: {query: str}
    Returns a list of search results.
    """
    args = step.get("args", {})
    query = str(args.get("query", ""))
    if not query:
        raise ValueError("web_search: 'query' is required.")
        
    import re
    # Dual-strategy search: Try Lite first, fallback to wide-harvest
    search_url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            
        results = []
        # Strategy 1: Targeted Lite regex
        # Lite version often uses <td><a class="result-link" ...>
        link_matches = re.findall(r'href="([^"]+)"[^>]*class="result-link"[^>]*>([^<]+)</a>', body)
        snippet_matches = re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', body, re.DOTALL)
        
        for i in range(min(len(link_matches), len(snippet_matches))):
            link = link_matches[i][0]
            if link.startswith("//"): link = "https:" + link
            title = re.sub(r'<[^>]+>', '', link_matches[i][1]).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet_matches[i]).strip()
            results.append({"title": title, "link": link, "snippet": snippet})
        
        # Strategy 2: Wide-harvest (if Strategy 1 got < 3 results)
        if len(results) < 3:
            # Just look for any external links that look like search results
            wide_matches = re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>([^<]{5,})</a>', body)
            for m in wide_matches:
                link, title = m.groups()
                if "http" in link and "duckduckgo" not in link and "bing" not in link:
                    # Avoid duplicated links
                    if not any(r["link"] == link for r in results):
                        results.append({"title": title.strip(), "link": link, "snippet": ""})
                if len(results) >= 8: break

        print(f"[Web] Search for '{query}' returned {len(results)} results.")
        return results
    except Exception as e:
        print(f"[Web] Search failed: {e}")
        
    # Last Resort: Minimal Google harvest
    try:
        print("[Web] Trying Google fallback...")
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        
        matches = re.finditer(r'<a href="/url\?q=([^&"]+)[^>]*><h3[^>]*>(.*?)</h3>', body)
        for m in matches:
            link = urllib.parse.unquote(m.group(1))
            if "http" in link and "google.com" not in link:
                results.append({
                    "title": re.sub(r'<[^>]+>', '', m.group(2)).strip(),
                    "link": link,
                    "snippet": "Result from Google search."
                })
            if len(results) >= 8: break
    except Exception:
        pass

    # If STILL empty, try Wikipedia (Unblockable and fast)
    if not results:
        try:
            print("[Web] Trying Wikipedia API fallback...")
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=5&format=json"
            with urllib.request.urlopen(wiki_url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # data format: [query, [titles], [snippets], [links]]
                for i in range(len(data[1])):
                    results.append({
                        "title": data[1][i],
                        "link": data[3][i],
                        "snippet": data[2][i]
                    })
        except Exception:
            pass

    print(f"[Web] Search for '{query}' returned {len(results)} results.")
    return results



# ── registration ───────────────────────────────────────────────────────────────

def setup_web_adapters(toolgate):
    toolgate.register_adapter("http_get",       http_get_adapter)
    toolgate.register_adapter("http_post",      http_post_adapter)
    toolgate.register_adapter("web_request",    http_get_adapter)  # alias
    toolgate.register_adapter("web_search",     web_search_adapter)
    toolgate.register_adapter("download_file",  download_file_adapter)
    print("[Adapters] Web/HTTP adapters registered.")
