# video-gen_free

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oscarneupane/video-gen_free/blob/main/notebooks/vidgen_gpu_server.ipynb)

Free text-to-video: the model runs on a **free cloud GPU** (Google Colab or
Kaggle notebook, or any GPU pod), and you drive it from **your own computer**
with a CLI or a local web UI. Finished videos are saved locally in `outputs/`.

```
  your computer                         free GPU (Colab / Kaggle / pod)
 ┌────────────────────────┐   HTTPS    ┌──────────────────────────────────┐
 │ python -m vidgen ui    │ ─────────► │ Cloudflare quick tunnel (free)   │
 │  localhost:7860        │  token     │  └► vidgen.server  (FastAPI)     │
 │  saves → outputs/*.mp4 │ ◄───────── │      └► diffusers: Wan / LTX     │
 └────────────────────────┘   mp4      └──────────────────────────────────┘
```

No paid API, no account besides Google/Kaggle: the tunnel is a Cloudflare
*quick tunnel*, which needs no sign-up.

### 0. Get the code locally

```bash
git clone https://github.com/oscarneupane/video-gen_free.git
cd video-gen_free
pip install -r requirements.txt   # optional: python-dotenv
```

### 1. Start the GPU server (in the browser)

1. Click **Open in Colab** above, or open `notebooks/vidgen_gpu_server.ipynb` in
   [Google Colab](https://colab.research.google.com) (*File → Upload notebook*) or
   [Kaggle](https://www.kaggle.com/code) (*File → Import notebook*).
2. Turn on a GPU — Colab: *Runtime → Change runtime type → T4 GPU*;
   Kaggle: *Settings → Accelerator → GPU T4* and *Internet → On*.
3. (Only for a private fork) add a `GITHUB_TOKEN` secret with read access.
4. *Run all*. The last cell prints:

   ```
   URL   : https://<random>.trycloudflare.com
   TOKEN : <random>
   ```

   Leave that cell running — it *is* the server.

On any other GPU box (RunPod, Vast.ai, Lightning AI, …) clone the repo and run
`bash vidgen/server/start_pod.sh` instead.

### 2. Generate from your computer

Only the Python standard library is needed locally. `python-dotenv` is optional
(`pip install -r requirements.txt`) and lets the CLI read `.env`.

```bash
python -m vidgen ui                         # web UI → paste URL + token → Generate
python -m vidgen status                     # check the connection
python -m vidgen generate "a red fox running through fresh snow at sunrise, cinematic"
python -m vidgen generate "ocean waves at night" --model ltx --frames 97 --seed 42
```

The UI saves the URL/token to `.env` (`VIDGEN_URL`, `VIDGEN_TOKEN`) so the CLI
picks them up too. The tunnel URL changes each time the notebook restarts.

### Models

| key | model | default output | notes |
|---|---|---|---|
| `wan-1.3b` | Wan 2.1 T2V 1.3B | 832×480, 33 frames @ 16 fps | best quality on a free T4 (default) |
| `ltx` | LTX-Video 2B | 704×480, 65 frames @ 24 fps | fast; needs Kaggle's ~30 GB RAM |
| `modelscope` | ModelScope T2V 1.7B | 256×256, 16 frames @ 8 fps | tiny, for quick tests |

Sizes and frame counts are snapped to what each model accepts and capped so a
16 GB card doesn't run out of memory. On a T4, a default Wan clip takes roughly
5–10 minutes (the first run also downloads the weights). The UI and CLI poll a
job queue, so long renders don't time out.

### Free GPU limits to know

- **Colab free:** T4 16 GB, sessions end after a few idle-ish hours; ~12 GB RAM.
- **Kaggle:** T4 ×2 or P100, ~30 GPU hours/week, 30 GB RAM, phone-verified account.
- Everything is lost when the session ends — the client already saved your videos locally.

### Try it without a GPU

```bash
python -m vidgen.server.launch --fake --no-tunnel --port 8000   # terminal 1
VIDGEN_URL=http://localhost:8000 VIDGEN_TOKEN=<printed> python -m vidgen ui   # terminal 2
```

### Layout

| File | Role |
|---|---|
| `vidgen/specs.py` | model catalogue; validates and snaps generation settings |
| `vidgen/server/backends.py` | diffusers pipelines (Wan / LTX / ModelScope) + a no-GPU fake |
| `vidgen/server/app.py` | job-queue HTTP API with token auth |
| `vidgen/server/launch.py` | starts the API + Cloudflare tunnel, prints URL/token |
| `vidgen/client.py` | local client (stdlib only) |
| `vidgen/__main__.py` | CLI: `ui`, `generate`, `status`, `models` |
| `vidgen/ui.py`, `vidgen/static/` | local web UI that proxies to the GPU server |
| `notebooks/vidgen_gpu_server.ipynb` | one-click Colab/Kaggle server |
| `tests/test_vidgen.py` | offline tests (fake backend, no GPU) |
