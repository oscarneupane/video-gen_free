"""Model catalogue + request normalisation, shared by the server and the client.

Pure Python — no torch/diffusers import — so the local side can validate
settings and the tests can run without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str            # Hugging Face repo id
    family: str          # which loader in vidgen.server.backends to use
    label: str
    notes: str
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    guidance: float
    dim_multiple: int    # width/height must be a multiple of this
    frame_step: int      # valid frame counts are frame_step * k + 1 (1 = anything)
    max_frames: int
    max_pixels: int      # width * height cap, keeps a 16 GB card out of OOM

    def public(self) -> dict:
        return asdict(self)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="wan-1.3b",
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            family="wan",
            label="Wan 2.1 T2V 1.3B",
            notes="Best quality/size trade-off. Fits a free T4 (~8 GB VRAM with offload).",
            width=832, height=480, frames=33, fps=16, steps=30, guidance=5.0,
            dim_multiple=16, frame_step=4, max_frames=81, max_pixels=832 * 480,
        ),
        ModelSpec(
            key="ltx",
            repo="Lightricks/LTX-Video",
            family="ltx",
            label="LTX-Video 2B",
            notes="Fast, 24 fps. Its T5 text encoder wants ~25 GB system RAM: use Kaggle, not free Colab.",
            width=704, height=480, frames=65, fps=24, steps=30, guidance=3.0,
            dim_multiple=32, frame_step=8, max_frames=161, max_pixels=768 * 512,
        ),
        ModelSpec(
            key="modelscope",
            repo="ali-vilab/text-to-video-ms-1.7b",
            family="modelscope",
            label="ModelScope T2V 1.7B",
            notes="Old and low-res (256px) but tiny and quick: good for testing the pipeline.",
            width=256, height=256, frames=16, fps=8, steps=25, guidance=9.0,
            dim_multiple=8, frame_step=1, max_frames=32, max_pixels=512 * 512,
        ),
    )
}

DEFAULT_MODEL = "wan-1.3b"

DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, watermark, text, static, "
    "worst quality, jpeg artifacts, extra limbs, ugly"
)


class SpecError(ValueError):
    """The request can't be turned into valid generation settings."""


def _snap(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _snap_frames(value: int, step: int, limit: int) -> int:
    if step <= 1:
        return max(1, min(value, limit))
    k = max(1, round((value - 1) / step))
    k = min(k, (limit - 1) // step)
    return step * k + 1


def normalize(request: dict) -> dict:
    """Fill defaults and snap sizes/frame counts to what the model accepts."""
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise SpecError("prompt is required")
    key = request.get("model") or DEFAULT_MODEL
    spec = MODELS.get(key)
    if spec is None:
        raise SpecError(f"unknown model {key!r}; choose one of {', '.join(MODELS)}")

    def pick(name, default, cast):
        raw = request.get(name)
        if raw is None or raw == "":
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            raise SpecError(f"{name} must be a number, got {raw!r}") from None

    width = _snap(pick("width", spec.width, int), spec.dim_multiple)
    height = _snap(pick("height", spec.height, int), spec.dim_multiple)
    if width * height > spec.max_pixels:
        # Shrink both sides by the same factor so the aspect ratio survives.
        scale = (spec.max_pixels / (width * height)) ** 0.5
        width = max(spec.dim_multiple, int(width * scale) // spec.dim_multiple * spec.dim_multiple)
        height = max(spec.dim_multiple, int(height * scale) // spec.dim_multiple * spec.dim_multiple)

    seed = pick("seed", None, int)
    negative = request.get("negative_prompt")
    return {
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE if negative is None else str(negative),
        "model": key,
        "width": width,
        "height": height,
        "frames": _snap_frames(pick("frames", spec.frames, int), spec.frame_step, spec.max_frames),
        "steps": max(1, min(pick("steps", spec.steps, int), 100)),
        "guidance": max(0.0, min(pick("guidance", spec.guidance, float), 20.0)),
        "fps": max(1, min(pick("fps", spec.fps, int), 60)),
        "seed": seed,
    }
