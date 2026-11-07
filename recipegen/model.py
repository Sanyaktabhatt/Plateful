"""U2: GPT-2 via KerasHub (the successor to KerasNLP), base and fine-tuned."""
from __future__ import annotations

import threading
import time

from . import config  # noqa: F401  (sets KERAS_BACKEND=tensorflow)
from .prompt_format import END, build_prompt, parse_output

_cache: dict[str, object] = {}
_lock = threading.Lock()


def finetuned_available() -> bool:
    return (config.GPT2_FINETUNED_DIR / "config.json").exists()


def _sampler(seed=None):
    import keras_hub
    return keras_hub.samplers.TopPSampler(p=0.9, temperature=0.8, seed=seed)


def load(which: str = "finetuned"):
    """which: 'base' or 'finetuned'. Models are cached after first load."""
    with _lock:
        if which in _cache:
            return _cache[which]
        import keras_hub
        if which == "finetuned":
            if not finetuned_available():
                raise FileNotFoundError(
                    f"No fine-tuned model at {config.GPT2_FINETUNED_DIR}. Run scripts/train_gpt2.py first.")
            lm = keras_hub.models.GPT2CausalLM.from_preset(str(config.GPT2_FINETUNED_DIR))
        else:
            lm = keras_hub.models.GPT2CausalLM.from_preset(config.GPT2_PRESET)
        lm.compile(sampler=_sampler())
        _cache[which] = lm
        return lm


def generate(ingredients: list[str], which: str = "finetuned", max_length: int = 384,
             seed: int | None = None) -> dict:
    import keras

    lm = load(which)
    if seed is not None:
        keras.utils.set_random_seed(seed)
        lm.compile(sampler=_sampler(seed))
    prompt = build_prompt(ingredients)
    t0 = time.time()
    out = lm.generate(prompt, max_length=max_length)
    elapsed = time.time() - t0
    if END in out:
        out = out.split(END, 1)[0] + END
    recipe = parse_output(out).to_dict()
    recipe.update(model=which, prompt=prompt, seconds=round(elapsed, 2))
    return recipe
