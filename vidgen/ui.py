"""Local web UI: a tiny stdlib server that proxies to the GPU server.

The browser only ever talks to localhost; this process holds the token, relays
jobs to the remote GPU, and saves finished videos into the local output folder.
"""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from vidgen.client import LocalConfig, PodClient, PodError, output_name
from vidgen.specs import MODELS, SpecError, normalize

STATIC = os.path.join(os.path.dirname(__file__), "static")


def save_env(path: str, values: dict[str, str]) -> None:
    """Upsert KEY=value lines in a .env file, leaving other lines alone."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    for key, value in values.items():
        entry = f"{key}={value}"
        for i, line in enumerate(lines):
            if re.match(rf"\s*{re.escape(key)}\s*=", line):
                lines[i] = entry
                break
        else:
            lines.append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class UIState:
    def __init__(self, cfg: LocalConfig, env_path: str = ".env") -> None:
        self.cfg = cfg
        self.env_path = env_path
        self.prompts: dict[str, str] = {}      # job id -> prompt (for file names)
        self.saved: dict[str, str] = {}        # job id -> local file name
        self.lock = threading.Lock()

    def client(self) -> PodClient:
        return PodClient(self.cfg.url, self.cfg.token)

    def fetch_video(self, job_id: str) -> str:
        """Download a finished job once; later calls return the cached name."""
        with self.lock:
            if job_id in self.saved:
                return self.saved[job_id]
            name = output_name(self.prompts.get(job_id, "video"), job_id)
            self.client().download(job_id, os.path.join(self.cfg.output_dir, name))
            self.saved[job_id] = name
            return name

    def gallery(self) -> list[dict]:
        folder = self.cfg.output_dir
        if not os.path.isdir(folder):
            return []
        items = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        items.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
        return [{"name": f, "url": f"/outputs/{f}"} for f in items[:60]]


def make_handler(state: UIState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "vidgen-ui"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        # -- helpers ------------------------------------------------------
        def _send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError:
                return {}

        def _send_file(self, path: str, ctype: str):
            size = os.path.getsize(path)
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                else:  # suffix range: last N bytes
                    start = max(0, size - int(match.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _guard(self, fn):
            try:
                fn()
            except SpecError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
            except PodError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # -- routes -------------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
            if path == "/api/config":
                return self._send_json({"url": state.cfg.url, "has_token": bool(state.cfg.token),
                                        "output_dir": os.path.abspath(state.cfg.output_dir)})
            if path == "/api/models":
                # Served locally so the form works before the GPU is connected.
                return self._send_json({"models": [m.public() for m in MODELS.values()]})
            if path == "/api/health":
                return self._guard(lambda: self._send_json(state.client().health()))
            if path == "/api/gallery":
                return self._send_json({"items": state.gallery()})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
            if m:
                def job():
                    data = state.client().job(m.group(1))
                    if data["status"] == "done":
                        data["video"] = "/outputs/" + state.fetch_video(m.group(1))
                    self._send_json(data)
                return self._guard(job)
            if path.startswith("/outputs/"):
                name = os.path.basename(unquote(path[len("/outputs/"):]))
                full = os.path.join(state.cfg.output_dir, name)
                if name.endswith(".mp4") and os.path.isfile(full):
                    return self._send_file(full, "video/mp4")
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/config":
                state.cfg.url = str(body.get("url", state.cfg.url)).strip().rstrip("/")
                if body.get("token"):
                    state.cfg.token = str(body["token"]).strip()
                if body.get("remember"):
                    save_env(state.env_path, {"VIDGEN_URL": state.cfg.url, "VIDGEN_TOKEN": state.cfg.token})
                return self._send_json({"url": state.cfg.url, "has_token": bool(state.cfg.token)})
            if path == "/api/generate":
                def generate():
                    settings = normalize(body)
                    job = state.client().submit(settings)
                    state.prompts[job["id"]] = settings["prompt"]
                    self._send_json(job)
                return self._guard(generate)
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def make_server(cfg: LocalConfig, port: int = 7860, env_path: str = ".env") -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(UIState(cfg, env_path)))


def serve(cfg: LocalConfig, port: int = 7860, open_browser: bool = True) -> None:
    httpd = make_server(cfg, port)
    url = f"http://127.0.0.1:{port}"
    print(f"vidgen UI on {url}  (videos saved to {os.path.abspath(cfg.output_dir)})")
    if not cfg.url:
        print("no GPU server configured yet: paste the URL + token from the notebook into the UI")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
