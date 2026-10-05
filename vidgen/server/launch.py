"""Start the GPU server and (by default) a free Cloudflare quick tunnel to it.

Run this on the GPU machine — a Colab/Kaggle notebook cell or a pod shell:

    python -m vidgen.server.launch                  # serve + tunnel, preload Wan
    python -m vidgen.server.launch --model ltx      # preload a different model
    python -m vidgen.server.launch --no-tunnel      # pod with its own public port
    python -m vidgen.server.launch --fake           # no GPU: synthetic clips

It prints the public URL and access token to paste into your local machine.
The quick tunnel needs no Cloudflare account; the URL changes every launch.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request

from vidgen.specs import DEFAULT_MODEL, MODELS

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
TUNNEL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def cloudflared_binary() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    if platform.system() != "Linux":
        sys.exit("cloudflared not found: install it from "
                 "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine(), "amd64")
    path = os.path.expanduser("~/.cache/vidgen/cloudflared")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading cloudflared ...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), path)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def start_tunnel(port: int, timeout: float = 60) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [cloudflared_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    found: list[str] = []

    def pump() -> None:
        # Keep draining output for the tunnel's lifetime so its pipe never fills.
        for line in proc.stdout:
            if not found:
                match = TUNNEL_RE.search(line)
                if match:
                    found.append(match.group(0))

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + timeout
    while not found and time.time() < deadline and proc.poll() is None:
        time.sleep(0.2)
    if not found:
        proc.kill()
        raise RuntimeError("cloudflared did not report a tunnel URL; is outbound internet enabled?")
    return proc, found[0]


def wait_until_up(port: int, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server did not start")


def banner(url: str, token: str) -> str:
    rule = "=" * 72
    return "\n".join([
        rule,
        "  vidgen GPU server is live",
        rule,
        f"  URL   : {url}",
        f"  TOKEN : {token}",
        "",
        "  On your own computer, put these in .env (in your video-gen_free folder):",
        f"    VIDGEN_URL={url}",
        f"    VIDGEN_TOKEN={token}",
        "  then run:  python -m vidgen ui      (or: python -m vidgen generate \"...\")",
        "",
        "  Keep this cell/terminal running. Stopping it shuts the server down.",
        rule,
    ])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.getenv("VIDGEN_PORT", "8000")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default=os.getenv("VIDGEN_PRELOAD", DEFAULT_MODEL),
                    help=f"model to load at start ({', '.join(MODELS)}, or 'none')")
    ap.add_argument("--token", default=os.getenv("VIDGEN_TOKEN", ""),
                    help="access token (default: a random one is generated)")
    ap.add_argument("--output-dir", default=os.getenv("VIDGEN_SERVER_OUTPUT", "vidgen_outputs"))
    ap.add_argument("--no-tunnel", action="store_true", help="don't open a Cloudflare tunnel")
    ap.add_argument("--fake", action="store_true", help="no GPU: render synthetic clips")
    args = ap.parse_args(argv)

    import uvicorn

    from vidgen.server.app import create_app
    from vidgen.server.backends import DiffusersBackend, FakeBackend

    token = args.token or secrets.token_urlsafe(18)
    backend = FakeBackend() if args.fake else DiffusersBackend()
    print(f"backend: {backend.info()}", flush=True)

    if not args.fake and args.model != "none":
        if args.model not in MODELS:
            sys.exit(f"unknown model {args.model!r}; choose one of {', '.join(MODELS)} or 'none'")
        print(f"loading {MODELS[args.model].label} (first run downloads the weights) ...", flush=True)
        backend.load(args.model)

    app = create_app(backend, token, args.output_dir)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    wait_until_up(args.port)

    tunnel = None
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"
    if not args.no_tunnel:
        tunnel, url = start_tunnel(args.port)
    print(banner(url, token), flush=True)

    try:
        while True:
            time.sleep(5)
            if tunnel is not None and tunnel.poll() is not None:
                print("tunnel died; restarting ...", flush=True)
                tunnel, url = start_tunnel(args.port)
                print(banner(url, token), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel is not None:
            tunnel.terminate()
        server.should_exit = True


if __name__ == "__main__":
    main()
