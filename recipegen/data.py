"""Download and clean RecipeNLG (via the corbt/all-recipes parquet mirror)."""
from __future__ import annotations

import json
import random
import re

from .config import DATA_DIR, RECIPES_JSONL
from .ingredients import parse_line
from .prompt_format import build_training_text

HF_REPO = "corbt/all-recipes"
HF_SHARD = "data/train-00000-of-00004-237b1b1141fdcfa1.parquet"


def parse_raw(text: str) -> dict | None:
    """'Title\\n\\nIngredients:\\n- ...\\n\\nDirections:\\n- ...' -> dict."""
    m = re.match(r"(.*?)\n+Ingredients:\n(.*?)\n+Directions:\n(.*)", text, re.S)
    if not m:
        return None
    title = m.group(1).strip()
    ings = [l.strip()[2:].strip() for l in m.group(2).splitlines() if l.strip().startswith("- ")]
    dirs = [l.strip()[2:].strip() for l in m.group(3).splitlines() if l.strip().startswith("- ")]
    if not title or not ings or not dirs:
        return None
    return {"title": title, "ingredient_lines": ings, "directions": dirs}


def keep(r: dict) -> bool:
    n_i, n_d = len(r["ingredient_lines"]), len(r["directions"])
    chars = len(r["title"]) + sum(map(len, r["ingredient_lines"])) + sum(map(len, r["directions"]))
    if not (3 <= n_i <= 14 and 1 <= n_d <= 12 and 120 <= chars <= 1100):
        return False
    if not re.search(r"[a-zA-Z]", r["title"]) or len(r["title"]) > 70:
        return False
    # skip recipes that are mostly non-ASCII junk
    return sum(c.isascii() for c in r["title"] + r["directions"][0]) / max(1, len(r["title"] + r["directions"][0])) > 0.95


def prepare(n: int = 8000, seed: int = 42) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_SHARD, repo_type="dataset")
    df = pd.read_parquet(path)
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = 0
    with open(RECIPES_JSONL, "w") as f:
        for i in idx:
            r = parse_raw(df.iloc[i]["input"])
            if not r or not keep(r):
                continue
            r["title"] = r["title"].title() if r["title"].isupper() else r["title"]
            names = [p.name for p in map(parse_line, r["ingredient_lines"]) if p.name]
            # drop pantry noise from the *prompt* so users aren't forced to list it
            core = [n for n in names if n not in ("salt", "pepper", "salt and pepper", "water")]
            if len(core) < 2:
                continue
            r["core_ingredients"] = core
            f.write(json.dumps(r) + "\n")
            out += 1
            if out >= n:
                break
    return out


def load_recipes() -> list[dict]:
    with open(RECIPES_JSONL) as f:
        return [json.loads(l) for l in f]


def training_texts(recipes: list[dict], seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for r in recipes:
        # Show the model a random subset of ingredients sometimes, so it learns
        # to fill in what the user forgot (salt, oil, ...).
        core = list(r["core_ingredients"])
        if len(core) > 4 and rng.random() < 0.3:
            core = rng.sample(core, rng.randint(3, len(core) - 1))
        texts.append(build_training_text(r["title"], core, r["ingredient_lines"], r["directions"], rng))
    return texts
