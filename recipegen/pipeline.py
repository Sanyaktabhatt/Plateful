"""End to end: user text -> NER -> GPT-2 recipe -> safety checks (image is separate)."""
from __future__ import annotations

from . import model, ner, safety


def run(user_text: str, which: str = "finetuned", seed: int | None = None, avoid: list[str] | None = None) -> dict:
    tagged, engine = ner.tag_user_input(user_text)
    names = [t["name"] for t in tagged if t["name"]]
    if not names:
        raise ValueError("No ingredients found in the input.")

    # Quantities the *user* typed are checked too ("5 cups salt").
    input_check = safety.analyze([t["line"] for t in tagged])

    models = ["base", "finetuned"] if which == "both" else [which]
    recipes = []
    for m in models:
        r = model.generate(names, which=m, seed=seed)
        check = safety.analyze(r["ingredients"] or [], r["directions"])
        # allergens from the user's own list count as well
        allergens = {k: list(v) for k, v in check["allergens"].items()}
        for k, v in input_check["allergens"].items():
            allergens.setdefault(k, [])
            allergens[k] += [x for x in v if x not in allergens[k]]
        avoid_hits = sorted(a for a in (avoid or []) if a in allergens)
        r["safety"] = {
            "allergens": allergens,
            "quantity_flags": check["quantity_flags"],
            "avoid_conflicts": avoid_hits,
        }
        recipes.append(r)

    return {
        "input": {"text": user_text, "tagged": tagged, "ner_engine": engine, "names": names,
                  "quantity_flags": input_check["quantity_flags"]},
        "recipes": recipes,
    }
