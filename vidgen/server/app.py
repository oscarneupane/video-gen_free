"""HTTP API for the GPU box.

Generation takes minutes and Cloudflare quick tunnels cut requests at ~100 s, so
the API is job-based: POST a job, poll its status, then download the video.
Jobs run one at a time on a single worker thread (one GPU, one pipeline).

    GET  /health             -> GPU info, loaded model, queue length
    GET  /models             -> model catalogue with defaults
    POST /jobs               -> {"id": ...}         body: generation settings
    GET  /jobs               -> recent jobs
    GET  /jobs/{id}          -> status / progress / error
    GET  /jobs/{id}/video    -> the mp4

Every route requires ``Authorization: Bearer <token>`` because the tunnel URL
is public.
"""

from __future__ import annotations

import hmac
import os
import queue
import threading
import time
import traceback
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from vidgen import __version__
from vidgen.specs import MODELS, SpecError, normalize

MAX_JOBS_KEPT = 50


class JobRunner:
    def __init__(self, backend, output_dir: str) -> None:
        self.backend = backend
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._work, name="vidgen-worker", daemon=True).start()

    def submit(self, settings: dict) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12],
            "status": "queued",
            "settings": settings,
            "progress": 0.0,
            "step": 0,
            "total_steps": settings["steps"],
            "error": None,
            "created": time.time(),
            "started": None,
            "finished": None,
        }
        with self._lock:
            self.jobs[job["id"]] = job
            self._prune()
        self._queue.put(job["id"])
        return job

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def video_path(self, job_id: str) -> str:
        return os.path.join(self.output_dir, f"{job_id}.mp4")

    def queued(self) -> int:
        return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "running"))

    def _prune(self) -> None:
        finished = [j for j in self.jobs.values() if j["status"] in ("done", "error")]
        finished.sort(key=lambda j: j["created"])
        for old in finished[: max(0, len(self.jobs) - MAX_JOBS_KEPT)]:
            self.jobs.pop(old["id"], None)
            try:
                os.remove(self.video_path(old["id"]))
            except OSError:
                pass

    def _work(self) -> None:
        while True:
            job = self.jobs.get(self._queue.get())
            if job is None:
                continue
            job["status"], job["started"] = "running", time.time()

            def on_progress(step: int, total: int, job=job) -> None:
                job["step"], job["total_steps"] = step, total
                job["progress"] = round(min(step / max(total, 1), 1.0), 3)

            try:
                self.backend.generate(job["settings"], self.video_path(job["id"]), on_progress)
                job["status"], job["progress"] = "done", 1.0
            except Exception as exc:  # report it to the client instead of killing the worker
                traceback.print_exc()
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                if "out of memory" in str(exc).lower():
                    job["error"] += " — try fewer frames or a smaller size."
            finally:
                job["finished"] = time.time()


def public_job(job: dict) -> dict:
    out = dict(job)
    end = job["finished"] or time.time()
    out["elapsed"] = round(end - job["started"], 1) if job["started"] else 0.0
    return out


def create_app(backend, token: str, output_dir: str = "vidgen_outputs") -> FastAPI:
    if not token:
        raise ValueError("a token is required: the tunnel URL is public")
    runner = JobRunner(backend, output_dir)
    app = FastAPI(title="vidgen GPU server", version=__version__)
    app.state.runner = runner

    def auth(authorization: str = Header(default="")) -> None:
        supplied = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="bad or missing token")

    guarded = [Depends(auth)]

    @app.get("/health", dependencies=guarded)
    def health():
        return {"ok": True, "version": __version__, "queue": runner.queued(), **backend.info()}

    @app.get("/models", dependencies=guarded)
    def models():
        return {"models": [m.public() for m in MODELS.values()]}

    @app.post("/jobs", dependencies=guarded)
    def create_job(body: dict):
        try:
            settings = normalize(body)
        except SpecError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return public_job(runner.submit(settings))

    @app.get("/jobs", dependencies=guarded)
    def list_jobs():
        jobs = sorted(runner.jobs.values(), key=lambda j: j["created"], reverse=True)
        return {"jobs": [public_job(j) for j in jobs]}

    @app.get("/jobs/{job_id}", dependencies=guarded)
    def get_job(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return public_job(job)

    @app.get("/jobs/{job_id}/video", dependencies=guarded)
    def get_video(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"job is {job['status']}")
        return FileResponse(runner.video_path(job_id), media_type="video/mp4",
                            filename=f"vidgen-{job_id}.mp4")

    return app
