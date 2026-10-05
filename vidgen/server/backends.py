"""Generation backends.

``DiffusersBackend`` is the real one (needs a CUDA GPU, torch and diffusers).
``FakeBackend`` writes a synthetic clip without a GPU so the whole loop — tunnel,
client, UI — can be tried and tested on any machine.

Both expose ``info() -> dict`` and
``generate(settings, out_path, on_progress(step, total))``.
"""

from __future__ import annotations

import gc
import inspect
import os
import threading
import time

from vidgen.specs import MODELS


class DiffusersBackend:
    """Keeps one pipeline resident; switching models unloads the previous one."""

    def __init__(self) -> None:
        import torch  # deferred: the local machine never imports this module path

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU found. In Colab: Runtime > Change runtime type > T4 GPU. "
                "In Kaggle: Settings > Accelerator > GPU T4. (Or run with --fake.)"
            )
        self.torch = torch
        props = torch.cuda.get_device_properties(0)
        self.gpu_name = props.name
        self.vram_gb = props.total_memory / 1024**3
        override = os.getenv("VIDGEN_DTYPE", "").lower()
        if override in {"fp16", "float16"}:
            self.dtype = torch.float16
        elif override in {"bf16", "bfloat16"}:
            self.dtype = torch.bfloat16
        else:
            # Ampere+ has native bf16; older cards (T4, P100) run fp16 much faster.
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self._pipe = None
        self._key: str | None = None
        self._lock = threading.Lock()

    def info(self) -> dict:
        return {
            "backend": "diffusers",
            "gpu": self.gpu_name,
            "vram_gb": round(self.vram_gb, 1),
            "dtype": str(self.dtype).replace("torch.", ""),
            "loaded": self._key,
        }

    # -- loading ----------------------------------------------------------
    def load(self, key: str) -> None:
        with self._lock:
            if self._key == key:
                return
            self._unload()
            spec = MODELS[key]
            pipe = getattr(self, f"_load_{spec.family}")(spec.repo)
            # Below ~20 GB, stream sub-models to the GPU on demand instead of
            # keeping all of them resident. Slower, but it's what makes a T4 work.
            if self.vram_gb < 20:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                try:
                    vae.enable_tiling()
                except Exception:  # not every VAE implements it in every diffusers version
                    pass
            self._pipe, self._key = pipe, key

    def _unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe, self._key = None, None
        gc.collect()
        self.torch.cuda.empty_cache()

    def _load_wan(self, repo: str):
        from diffusers import AutoencoderKLWan, WanPipeline

        # The Wan VAE is unstable in half precision, so it stays in fp32.
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=self.torch.float32)
        return WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=self.dtype)

    def _load_ltx(self, repo: str):
        from diffusers import LTXPipeline

        return LTXPipeline.from_pretrained(repo, torch_dtype=self.dtype)

    def _load_modelscope(self, repo: str):
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(repo, torch_dtype=self.torch.float16, variant="fp16")

    # -- generation -------------------------------------------------------
    def generate(self, s: dict, out_path: str, on_progress) -> None:
        from diffusers.utils import export_to_video

        self.load(s["model"])
        pipe = self._pipe
        kwargs = dict(
            prompt=s["prompt"],
            negative_prompt=s["negative_prompt"] or None,
            width=s["width"],
            height=s["height"],
            num_frames=s["frames"],
            num_inference_steps=s["steps"],
            guidance_scale=s["guidance"],
        )
        if s.get("seed") is not None:
            kwargs["generator"] = self.torch.Generator(device="cpu").manual_seed(s["seed"])

        params = inspect.signature(pipe.__call__).parameters
        total = s["steps"]
        if "callback_on_step_end" in params:
            def _step_end(_pipe, step, _timestep, cb_kwargs):
                on_progress(step + 1, total)
                return cb_kwargs
            kwargs["callback_on_step_end"] = _step_end
        elif "callback" in params:  # older pipelines (ModelScope)
            kwargs["callback"] = lambda step, _t, _latents: on_progress(step + 1, total)
            kwargs["callback_steps"] = 1

        with self.torch.inference_mode():
            frames = pipe(**kwargs).frames[0]
        export_to_video(frames, out_path, fps=s["fps"])
        on_progress(total, total)


class FakeBackend:
    """No-GPU stand-in: renders a moving gradient so the pipeline is visible."""

    def __init__(self, step_delay: float = 0.05) -> None:
        self.step_delay = step_delay
        self._key: str | None = None

    def info(self) -> dict:
        return {"backend": "fake", "gpu": None, "vram_gb": 0, "dtype": None, "loaded": self._key}

    def generate(self, s: dict, out_path: str, on_progress) -> None:
        self._key = s["model"]
        for step in range(s["steps"]):
            time.sleep(self.step_delay)
            on_progress(step + 1, s["steps"])
        try:
            import imageio.v2 as imageio
            import numpy as np

            h, w = s["height"], s["width"]
            ys, xs = np.mgrid[0:h, 0:w]
            frames = []
            for i in range(s["frames"]):
                t = i / max(1, s["frames"] - 1)
                r = (xs / w * 255 + t * 255) % 256
                g = (ys / h * 255) % 256
                b = np.full_like(r, int(128 + 127 * t))
                frames.append(np.stack([r, g, b], axis=-1).astype("uint8"))
            imageio.mimwrite(out_path, frames, fps=s["fps"])
        except Exception:
            # numpy/imageio/ffmpeg missing: still produce a file so the flow completes.
            with open(out_path, "wb") as fh:
                fh.write(b"vidgen fake video\n")
