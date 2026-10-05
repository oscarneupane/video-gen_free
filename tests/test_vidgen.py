"""vidgen tests — no GPU, no network: the server runs the fake backend locally.

Run:  python -m pytest -q
"""

import json
import socket
import threading
import time
import urllib.request

import pytest

from vidgen.client import LocalConfig, PodClient, PodError
from vidgen.specs import DEFAULT_NEGATIVE, SpecError, normalize
from vidgen.ui import make_server, save_env


def test_normalize_fills_defaults_and_snaps():
    s = normalize({"prompt": "  a cat  "})
    assert s["prompt"] == "a cat"
    assert (s["model"], s["width"], s["height"], s["frames"]) == ("wan-1.3b", 832, 480, 33)
    assert s["negative_prompt"] == DEFAULT_NEGATIVE

    s = normalize({"prompt": "x", "model": "ltx", "width": 700, "height": 470, "frames": 100})
    assert s["width"] % 32 == 0 and s["height"] % 32 == 0
    assert (s["frames"] - 1) % 8 == 0

    s = normalize({"prompt": "x", "frames": 999, "width": 1920, "height": 1080, "negative_prompt": ""})
    assert s["frames"] == 81                         # capped at the model max, still 4k+1
    assert s["width"] * s["height"] <= 832 * 480     # pixel cap, aspect kept
    assert s["width"] > s["height"]
    assert s["negative_prompt"] == ""                # explicit empty disables the default


@pytest.mark.parametrize("bad", [{}, {"prompt": "x", "model": "nope"}, {"prompt": "x", "frames": "lots"}])
def test_normalize_rejects(bad):
    with pytest.raises(SpecError):
        normalize(bad)


def test_save_env_upserts(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=abc\nVIDGEN_URL=old\n")
    save_env(str(env), {"VIDGEN_URL": "https://new", "VIDGEN_TOKEN": "t"})
    assert env.read_text() == "ANTHROPIC_API_KEY=abc\nVIDGEN_URL=https://new\nVIDGEN_TOKEN=t\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gpu_server(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from vidgen.server.app import create_app
    from vidgen.server.backends import FakeBackend

    port = _free_port()
    app = create_app(FakeBackend(step_delay=0), "secret", str(tmp_path / "server"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def test_client_end_to_end(gpu_server, tmp_path):
    client = PodClient(gpu_server, "secret")
    health = client.health()
    assert health["ok"] and health["backend"] == "fake"
    assert {m["key"] for m in client.models()} >= {"wan-1.3b", "ltx", "modelscope"}

    seen = []
    settings = normalize({"prompt": "a fox", "model": "modelscope", "steps": 3, "frames": 4})
    job, path = client.generate(settings, str(tmp_path / "out" / "fox.mp4"), on_update=seen.append, poll=0.05)
    assert job["status"] == "done" and job["progress"] == 1.0
    assert (tmp_path / "out" / "fox.mp4").stat().st_size > 0
    assert seen[-1]["status"] == "done"

    with pytest.raises(PodError, match="token"):
        PodClient(gpu_server, "wrong").health()
    with pytest.raises(PodError, match="422"):
        client.submit({"prompt": ""})


def test_client_reports_unreachable():
    with pytest.raises(PodError, match="can't reach"):
        PodClient(f"http://127.0.0.1:{_free_port()}", "t", timeout=2).health()


def test_ui_proxies_and_saves(gpu_server, tmp_path):
    cfg = LocalConfig(url=gpu_server, token="secret", output_dir=str(tmp_path / "videos"))
    httpd = make_server(cfg, port=0, env_path=str(tmp_path / ".env"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None, headers=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                     headers=headers or {})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers

    try:
        assert b"vidgen" in call("/")[1]
        assert json.loads(call("/api/health")[1])["backend"] == "fake"

        job = json.loads(call("/api/generate", {"prompt": "sunset over the sea", "model": "modelscope",
                                                "steps": 2, "frames": 4})[1])
        for _ in range(200):
            data = json.loads(call(f"/api/jobs/{job['id']}")[1])
            if data["status"] == "done":
                break
            time.sleep(0.05)
        assert data["video"].startswith("/outputs/") and "sunset-over-the-sea" in data["video"]

        gallery = json.loads(call("/api/gallery")[1])["items"]
        assert [g["url"] for g in gallery] == [data["video"]]
        status, chunk, headers = call(data["video"], headers={"Range": "bytes=0-3"})
        assert status == 206 and len(chunk) == 4 and headers["Content-Range"].startswith("bytes 0-3/")

        call("/api/config", {"url": gpu_server, "token": "secret", "remember": True})
        assert "VIDGEN_TOKEN=secret" in (tmp_path / ".env").read_text()
    finally:
        httpd.shutdown()
        httpd.server_close()
