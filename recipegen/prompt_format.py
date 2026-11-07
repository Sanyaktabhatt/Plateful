"""U1: the ingredient-to-recipe prompt format.

A training example looks like this:

    <|recipe|>
    <|ingredients|> chicken thighs; garlic; lemon; rosemary
    <|title|> Lemon Rosemary Chicken
    <|ingredient_list|>
    - 2 lb. chicken thighs
    - 4 cloves garlic, minced
    <|directions|>
    1. Heat the oven to 425°.
    2. ...
    <|end|>

Design decisions:
* The conditioning line holds bare ingredient *names* (no quantities), shuffled,
  because that is what a user types. The model must invent quantities and order.
* Each section has its own marker so the output can be parsed back into a
  structured recipe and so generation can stop cleanly at <|end|>.
* Markers are plain text. GPT-2 splits them into a few BPE tokens, which is fine:
  they are consistent, so the model learns them within a few hundred steps, and
  we avoid resizing the embedding matrix.
* At inference time the prompt ends at "<|title|>", so the model writes the
  title first. A title commits it to a dish, which keeps the rest coherent.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict

START = "<|recipe|>"
INGREDIENTS = "<|ingredients|>"
TITLE = "<|title|>"
INGREDIENT_LIST = "<|ingredient_list|>"
DIRECTIONS = "<|directions|>"
END = "<|end|>"


@dataclass
class Recipe:
    title: str = ""
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    raw: str = ""
    complete: bool = False

    def to_dict(self):
        return asdict(self)


def _clean_names(names):
    out = []
    for n in names:
        n = re.sub(r"\s+", " ", str(n)).strip(" ;,.-").lower()
        if n and n not in out:
            out.append(n)
    return out


def build_prompt(ingredient_names: list[str]) -> str:
    """Inference prompt: the model continues from the title marker."""
    names = "; ".join(_clean_names(ingredient_names))
    return f"{START}\n{INGREDIENTS} {names}\n{TITLE}"


def build_training_text(title, core_names, ingredient_lines, directions, rng=None) -> str:
    rng = rng or random
    names = _clean_names(core_names)
    rng.shuffle(names)
    lines = "\n".join(f"- {l.strip()}" for l in ingredient_lines)
    steps = "\n".join(f"{i}. {s.strip()}" for i, s in enumerate(directions, 1))
    return (
        f"{START}\n{INGREDIENTS} {'; '.join(names)}\n{TITLE} {title.strip()}\n"
        f"{INGREDIENT_LIST}\n{lines}\n{DIRECTIONS}\n{steps}\n{END}"
    )


def parse_output(text: str) -> Recipe:
    """Parse generated text (prompt included or not) back into a Recipe."""
    r = Recipe(raw=text)
    body = text
    if TITLE in body:
        body = body.split(TITLE, 1)[1]
    r.complete = END in body
    body = body.split(END, 1)[0].split("<|endoftext|>", 1)[0]
    continuation = body
    # Base GPT-2 often starts a new recipe instead of finishing this one.
    body = body.split(START, 1)[0]

    title_part, _, rest = body.partition(INGREDIENT_LIST)
    r.title = title_part.strip().split("\n")[0].strip()[:120]
    # RecipeNLG titles are title-cased naively ("Mom'S"); the model copies that.
    r.title = re.sub(r"(\w)'S\b", r"\1's", r.title)
    ing_part, _, dir_part = rest.partition(DIRECTIONS)

    for line in ing_part.splitlines():
        line = line.strip()
        if line.startswith("-"):
            line = line.lstrip("- ").strip()
            if line:
                r.ingredients.append(line)
    for line in dir_part.splitlines():
        line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if line:
            r.directions.append(line)

    if not r.ingredients and not r.directions:
        # Unstructured output (typical of the base model): keep the whole
        # continuation as prose, so the comparison shows what it really wrote.
        prose = re.sub(r"<\|[a-z_]+\|>", " ", continuation)
        lines = [re.sub(r"\s+", " ", p).strip() for p in prose.split("\n")]
        r.directions = [p for p in lines if p][:12]
    return r
