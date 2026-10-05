"""vidgen CLI — drive the remote GPU from your machine.

    python -m vidgen ui                              # local web UI on :7860
    python -m vidgen generate "a red fox running through snow, cinematic"
    python -m vidgen generate "..." --model ltx --frames 97 --seed 7
    python -m vidgen status                          # is the GPU server up?
    python -m vidgen models                          # what can it run?

URL and token come from VIDGEN_URL / VIDGEN_TOKEN in .env (the notebook prints
both), or --url / --token.
"""

from __future__ import annotations

import argparse
import os
import sys

from vidgen.client import PodClient, PodError, load_config, output_name
from vidgen.specs import DEFAULT_MODEL, MODELS, SpecError, normalize


def _cmd_status(client: PodClient, _args) -> None:
    h = client.health()
    gpu = f"{h['gpu']} ({h['vram_gb']} GB, {h['dtype']})" if h.get("gpu") else h["backend"]
    print(f"connected to {client.url}")
    print(f"  gpu    : {gpu}")
    print(f"  loaded : {h.get('loaded') or '(none yet)'}")
    print(f"  queue  : {h['queue']} job(s)")


def _cmd_models(client: PodClient, _args) -> None:
    for m in client.models():
        print(f"  {m['key']:<11} {m['label']}  —  {m['width']}x{m['height']}, "
              f"{m['frames']} frames @ {m['fps']} fps")
        print(f"  {'':<11} {m['notes']}")


def _cmd_generate(client: PodClient, args) -> None:
    request = {k: v for k, v in {
        "prompt": " ".join(args.prompt),
        "negative_prompt": args.negative,
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "steps": args.steps,
        "guidance": args.guidance,
        "fps": args.fps,
        "seed": args.seed,
    }.items() if v is not None}
    settings = normalize(request)  # fail fast locally, and show what will run
    spec = MODELS[settings["model"]]
    secs = settings["frames"] / settings["fps"]
    print(f"{spec.label}: {settings['width']}x{settings['height']}, {settings['frames']} frames "
          f"(~{secs:.1f}s @ {settings['fps']} fps), {settings['steps']} steps")

    job = client.submit(settings)
    print(f"job {job['id']} queued")
    last = [None]

    def show(j: dict) -> None:
        line = (f"\r  {j['status']:<8} {int(j['progress'] * 100):>3}%  "
                f"step {j['step']}/{j['total_steps']}  {j['elapsed']:.0f}s")
        if line != last[0]:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = line

    try:
        job = client.wait(job["id"], on_update=show)
    finally:
        print()
    out = args.out or os.path.join(load_config().output_dir, output_name(settings["prompt"], job["id"]))
    client.download(job["id"], out)
    print(f"saved {out}  ({job['elapsed']:.0f}s on the GPU)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vidgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="GPU server URL (default: $VIDGEN_URL)")
    ap.add_argument("--token", help="GPU server token (default: $VIDGEN_TOKEN)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="open the local web UI")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--no-browser", action="store_true")

    sub.add_parser("status", help="check the GPU server")
    sub.add_parser("models", help="list models the server can run")

    gen = sub.add_parser("generate", help="make a video from a text prompt")
    gen.add_argument("prompt", nargs="+")
    gen.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    gen.add_argument("--negative", help="negative prompt ('' to disable the default)")
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--frames", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--guidance", type=float)
    gen.add_argument("--fps", type=int)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--out", help="output .mp4 path (default: outputs/<timestamp>-<prompt>.mp4)")

    args = ap.parse_args(argv)
    cfg = load_config(args.url, args.token)

    if args.cmd == "ui":
        from vidgen.ui import serve

        serve(cfg, port=args.port, open_browser=not args.no_browser)
        return

    try:
        client = PodClient(cfg.url, cfg.token)
        {"status": _cmd_status, "models": _cmd_models, "generate": _cmd_generate}[args.cmd](client, args)
    except (PodError, SpecError) as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted (the job keeps running on the GPU server)")


if __name__ == "__main__":
    main()
