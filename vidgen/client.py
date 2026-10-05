"""Local-side client for the GPU server. Standard library only."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional on the local side
    pass


class PodError(RuntimeError):
    """The GPU server is unreachable, rejected us, or the job failed."""


@dataclass
class LocalConfig:
    url: str
    token: str
    output_dir: str


def load_config(url: str | None = None, token: str | None = None) -> LocalConfig:
    return LocalConfig(
        url=(url or os.getenv("VIDGEN_URL", "")).strip().rstrip("/"),
        token=(token or os.getenv("VIDGEN_TOKEN", "")).strip(),
        output_dir=os.getenv("VIDGEN_OUTPUT_DIR", "outputs"),
    )


class PodClient:
    def __init__(self, url: str, token: str, timeout: float = 30) -> None:
        if not url:
            raise PodError("no GPU server URL: set VIDGEN_URL in .env (printed by the notebook) or pass --url")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        # Cloudflare's bot filter rejects the default Python user agent.
        req.add_header("User-Agent", "vidgen-client/0.1")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("detail", "")
            except Exception:
                detail = exc.reason
            if exc.code == 401:
                raise PodError("the GPU server rejected the token (check VIDGEN_TOKEN)") from None
            if exc.code in (502, 530):
                raise PodError("tunnel is up but the server behind it is gone — restart the notebook cell") from None
            raise PodError(f"{exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise PodError(f"can't reach {self.url} ({reason}); is the notebook still running?") from None

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        with self._request(method, path, body) as resp:
            return json.loads(resp.read().decode())

    def health(self) -> dict:
        return self._json("GET", "/health")

    def models(self) -> list[dict]:
        return self._json("GET", "/models")["models"]

    def submit(self, settings: dict) -> dict:
        return self._json("POST", "/jobs", settings)

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/jobs/{job_id}")

    def download(self, job_id: str, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".part"
        with self._request("GET", f"/jobs/{job_id}/video") as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        os.replace(tmp, path)
        return path

    def wait(self, job_id: str, on_update=None, poll: float = 3.0, timeout: float = 3 * 3600) -> dict:
        deadline = time.time() + timeout
        while True:
            job = self.job(job_id)
            if on_update:
                on_update(job)
            if job["status"] == "done":
                return job
            if job["status"] == "error":
                raise PodError(f"generation failed: {job['error']}")
            if time.time() > deadline:
                raise PodError(f"gave up waiting for job {job_id}")
            time.sleep(poll)

    def generate(self, settings: dict, out_path: str, on_update=None, poll: float = 3.0) -> tuple[dict, str]:
        """Submit, wait, download. Returns (final job, local path)."""
        job = self.submit(settings)
        job = self.wait(job["id"], on_update=on_update, poll=poll)
        return job, self.download(job["id"], out_path)


def output_name(prompt: str, job_id: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in prompt.lower())[:40].strip("-")
    slug = "-".join(filter(None, slug.split("-"))) or "video"
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{job_id[:6]}.mp4"
