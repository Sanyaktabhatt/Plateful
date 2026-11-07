"""U3 (part 2): dish photo from the generated recipe, using a diffusion model.

Default is SD-Turbo (Stable Diffusion distilled to 1–4 steps), run via
diffusers on Apple MPS, CUDA or CPU. The prompt is built from the recipe
title plus its main tagged ingredients, written as a food-photography brief.
"""
from __future__ import annotations

import base64
import io
import threading

from . import config

_pipe = None
_lock = threading.Lock()
_error: str | None = None


def available() -> bool:
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401  (CLIP text encoder)
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def build_image_prompt(title: str, ingredients: list[str]) -> str:
    main = ", ".join(ingredients[:4])
    dish = title or main
    return (f"{dish}, plated, made with {main}. Overhead food photography, natural window light, "
            "rustic ceramic plate on a linen tablecloth, shallow depth of field, appetizing, high detail")


NEGATIVE = "text, watermark, logo, hands, people, blurry, deformed, cartoon, illustration"


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load():
    global _pipe, _error
    with _lock:
        if _pipe is None and _error is None:
            try:
                import torch
                from diffusers import StableDiffusionPipeline  # SD-Turbo is SD 2.1 architecture
                device, dtype = _device()
                kw = {"torch_dtype": dtype}
                if dtype == torch.float16:
                    kw["variant"] = "fp16"
                pipe = StableDiffusionPipeline.from_pretrained(config.DIFFUSION_MODEL, **kw)
                _pipe = pipe.to(device)
                _pipe.set_progress_bar_config(disable=True)
            except Exception as e:  # missing deps, no network, out of memory...
                _error = f"{type(e).__name__}: {e}"
    return _pipe


def generate_image(title: str, ingredients: list[str], seed: int = 0, steps: int = 2) -> dict:
    """Returns {'image': data-URL or None, 'prompt': str, 'error': str|None}."""
    prompt = build_image_prompt(title, ingredients)
    if not available():
        return {"image": None, "prompt": prompt,
                "error": "diffusers/torch not installed (pip install -r requirements-image.txt)"}
    pipe = _load()
    if pipe is None:
        return {"image": None, "prompt": prompt, "error": _error}
    import torch
    gen = torch.Generator("cpu").manual_seed(seed)
    img = pipe(prompt=prompt, negative_prompt=NEGATIVE if steps > 1 else None,
               num_inference_steps=steps, guidance_scale=0.0, generator=gen,
               width=512, height=512).images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            "prompt": prompt, "error": None}
