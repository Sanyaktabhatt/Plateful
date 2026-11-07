"""Download one RecipeNLG shard and write a cleaned sample to data/recipes.jsonl."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipegen.config import RECIPES_JSONL  # noqa: E402
from recipegen.data import load_recipes, prepare, training_texts  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="number of recipes to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    count = prepare(args.n, args.seed)
    print(f"wrote {count} recipes to {RECIPES_JSONL}")
    print("\nexample training text:\n")
    print(training_texts(load_recipes()[:1])[0])
