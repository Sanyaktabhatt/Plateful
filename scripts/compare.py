"""Deliverable: base vs fine-tuned GPT-2 outputs for 5 ingredient sets.

Writes outputs/comparison.json (read by the app's Taste Test page) and
outputs/comparison.md (for the report). Same prompt and seed for both models.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recipegen  # noqa: E402,F401
from recipegen import config, model, safety  # noqa: E402


def score(r: dict, ingredients: list[str]) -> dict:
    """Simple structure checks so the comparison has numbers, not just vibes."""
    text = " ".join(r["ingredients"] + r["directions"]).lower()
    used = [i for i in ingredients if i.split()[-1].rstrip("s") in text]
    return {
        "has_title": bool(r["title"]),
        "ingredient_lines": len(r["ingredients"]),
        "steps": len(r["directions"]) if r["ingredients"] else 0,
        "finished_cleanly": r["complete"],
        "input_coverage": round(len(used) / len(ingredients), 2),
    }


def md_recipe(r):
    out = [f"**{r['title'] or '(no title)'}**", ""]
    if r["ingredients"]:
        out += [f"- {l}" for l in r["ingredients"]] + [""]
    out += [f"{i}. {s}" for i, s in enumerate(r["directions"], 1)] if r["ingredients"] else \
           ["> " + " ".join(r["directions"])[:600]]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse and re-score the saved raw outputs instead of generating")
    args = ap.parse_args()

    results = []
    if args.reparse:
        from recipegen.prompt_format import parse_output
        saved = json.loads(config.COMPARISON_JSON.read_text())
        args.seed = saved["seed"]
        for row in saved["sets"]:
            for which, old in row["outputs"].items():
                r = parse_output(old["raw"]).to_dict()
                r.update(model=which, prompt=old["prompt"], seconds=old["seconds"])
                r["scores"] = score(r, row["ingredients"])
                r["safety"] = safety.analyze(r["ingredients"], r["directions"])
                row["outputs"][which] = r
            results.append(row)
    for i, ings in enumerate([] if args.reparse else config.EVAL_INGREDIENT_SETS, 1):
        row = {"ingredients": ings, "outputs": {}}
        for which in ("base", "finetuned"):
            t0 = time.time()
            r = model.generate(ings, which=which, seed=args.seed + i)
            r["scores"] = score(r, ings)
            r["safety"] = safety.analyze(r["ingredients"], r["directions"])
            row["outputs"][which] = r
            print(f"[{i}] {which:9} {time.time() - t0:5.1f}s  {r['title']!r}  {r['scores']}")
        results.append(row)

    report = None
    if config.TRAINING_REPORT.exists():
        report = json.loads(config.TRAINING_REPORT.read_text())
    summary = {}
    for which in ("base", "finetuned"):
        s = [row["outputs"][which]["scores"] for row in results]
        summary[which] = {
            "structured_outputs": sum(x["ingredient_lines"] > 0 for x in s),
            "finished_cleanly": sum(x["finished_cleanly"] for x in s),
            "mean_input_coverage": round(sum(x["input_coverage"] for x in s) / len(s), 2),
            "val_perplexity": report[which]["val_perplexity"] if report else None,
        }

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    config.COMPARISON_JSON.write_text(json.dumps({"seed": args.seed, "summary": summary, "sets": results}, indent=2))

    md = ["# Base vs fine-tuned GPT-2: five ingredient sets", "",
          f"Same prompt format, sampler (top-p 0.9, temperature 0.8) and seed for both models.", "",
          "| | Base GPT-2 | Fine-tuned |", "|---|---|---|"]
    labels = {"structured_outputs": "Outputs with an ingredient list (of 5)",
              "finished_cleanly": "Stopped at `<|end|>` (of 5)",
              "mean_input_coverage": "Input ingredients used (mean)",
              "val_perplexity": "Validation perplexity"}
    for k, label in labels.items():
        b, f = summary["base"][k], summary["finetuned"][k]
        fmt = (lambda v: f"{v:.1f}") if k == "val_perplexity" and b else str
        md.append(f"| {label} | {fmt(b)} | {fmt(f)} |")
    for i, row in enumerate(results, 1):
        md += ["", "---", "", f"## {i}. {', '.join(row['ingredients'])}", "",
               "### Base GPT-2", "", md_recipe(row["outputs"]["base"]), "",
               "### Fine-tuned", "", md_recipe(row["outputs"]["finetuned"])]
        fl = row["outputs"]["finetuned"]["safety"]
        if fl["allergens"]:
            md += ["", f"*Allergens:* {', '.join(fl['allergens'])}"]
        for q in fl["quantity_flags"]:
            md += [f"*Flag ({q['level']}):* {q['line']}: {q['reason']}"]
    config.COMPARISON_MD.write_text("\n".join(md) + "\n")
    print("wrote", config.COMPARISON_JSON, "and", config.COMPARISON_MD)


if __name__ == "__main__":
    main()
