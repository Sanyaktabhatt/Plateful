# Larder: recipe generator fine-tuned in Keras (P45)

Type the ingredients you have. A GPT-2 model fine-tuned on RecipeNLG writes a full recipe
(title, measured ingredient list, numbered method), a diffusion model photographs the dish,
and a safety pass flags allergens and impossible quantities.

| Unit | What was built | Where |
|---|---|---|
| U1 | Ingredient-to-recipe prompt format with section markers | [recipegen/prompt_format.py](recipegen/prompt_format.py) |
| U2 | GPT-2 base (KerasHub, the successor to KerasNLP) fine-tuned on recipes, compared with the base model | [scripts/train_gpt2.py](scripts/train_gpt2.py), [scripts/compare.py](scripts/compare.py) |
| U3 | BERT token classifier tags QTY / UNIT / ING; SD-Turbo diffusion generates the dish photo | [recipegen/ner.py](recipegen/ner.py), [scripts/train_ner.py](scripts/train_ner.py), [recipegen/imagegen.py](recipegen/imagegen.py) |
| U4 | All training in TensorFlow 2: Keras 3 on the TF backend, `tf.data` pipelines, XLA-compiled `fit()` | both `train_*.py` scripts |
| U5 | Allergen warnings (US Big 9 + EU extras) and impossible / unusual quantity, temperature and time flags | [recipegen/safety.py](recipegen/safety.py) |

**Deliverables**
- Recipe app: [app/](app/), run with `python app/server.py`
- Base vs fine-tuned outputs for 5 ingredient sets: [outputs/comparison.md](outputs/comparison.md)
  (also shown on the app's *Taste Test* page)

## Results (this run)

| | Base GPT-2 | Fine-tuned |
|---|---|---|
| Validation perplexity (200 held-out recipes) | 6.5 | **2.7** |
| Outputs with an ingredient list, of 5 | 0 | **5** |
| Stopped cleanly at `<\|end\|>`, of 5 | 0 | **5** |
| Input ingredients used (mean) | 12% | **76%** |

Trained on 4,000 recipes, 500 steps, 72 min on an Apple M4 CPU with XLA. The BERT tagger
reaches 0.98 F1 on ingredient spans against held-out weak labels. Base GPT-2 never writes a
recipe: it loops on the markers and invents new ingredient lists. The fine-tuned model writes
plausible home-style recipes but still has quirks, like "1 (8 oz.) can rosemary" or vanilla
in a salmon dish. The safety pass exists for exactly that kind of output.

## Setup

TensorFlow needs Python 3.10–3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-image.txt     # optional: dish photos (torch + diffusers)
```

## Run the pipeline

```bash
python scripts/prepare_data.py --n 8000        # download 1 RecipeNLG shard, clean, sample → data/recipes.jsonl
python scripts/train_ner.py                    # BERT-tiny ingredient tagger, ~4 min on CPU
python scripts/train_gpt2.py --samples 4000    # GPT-2 fine-tune, ~70 min on an M4 CPU
python scripts/compare.py                      # base vs fine-tuned on the 5 sets → outputs/
python app/server.py                           # http://127.0.0.1:5045
```

The app degrades gracefully: without a fine-tuned model only the base model is offered;
without the BERT weights the rule tagger is used; without torch/diffusers the photo is
replaced by a placeholder.

## How it works

**Prompt format (U1).** Each recipe becomes:

```
<|recipe|>
<|ingredients|> lemon; chicken thighs; rosemary; garlic      ← names only, shuffled
<|title|> Lemon Rosemary Chicken                              ← generation starts here
<|ingredient_list|>
- 2 lb. chicken thighs
<|directions|>
1. Heat oven to 425°.
<|end|>
```

The conditioning line holds bare names because that is what users type. 30% of training
examples drop some ingredients from it, so the model learns to add staples you forgot. Recipes
longer than the 256-token window are dropped instead of truncated, so every example ends in
`<|end|>` and the model learns to stop.

**Fine-tuning (U2, U4).** `GPT2CausalLM.from_preset("gpt2_base_en")`, AdamW (lr 5e-5, linear
decay, weight decay 0.01), batch 8, 1 epoch over 4,000 recipes. Validation loss is measured on
the base model before training and on the fine-tuned model after, and both are written to
`models/gpt2_recipes/training_report.json`.

**Ingredient NER (U3).** RecipeNLG has no token labels, so the rule parser in
`recipegen/ingredients.py` produces weak BIO labels (QTY, UNIT, ING) over 30k ingredient lines.
BERT-tiny is fine-tuned on them with augmentation (quantities dropped, lowercased), so it copes
with how people type: "a handful of basil", "Spaghetti", "500g baby potatoes". Only ING spans
go into the prompt; amounts are kept for the quantity check. Reported F1 is measured against
the weak labels, so it shows agreement with the rules, not accuracy against human labels.

**Dish image (U3).** The recipe title plus the tagged ingredients become a food-photography
prompt for `stabilityai/sd-turbo` (2 steps, 512×512, about 4 s on Apple MPS).

**Safety (U5).** Runs on the user's list *and* the generated recipe, since the model adds
ingredients. Allergen keywords come with exceptions (coconut milk isn't dairy, nutmeg isn't a
nut, buckwheat isn't wheat). Quantities are converted to g or ml and checked against
per-ingredient limits (salt, leaveners, spices, extracts, saffron), general limits, and
zero/negative amounts. Directions are checked for oven temperatures and cooking times.

## Layout

```
recipegen/          library: prompt format, data, models, NER, diffusion, safety, pipeline
scripts/            prepare_data, train_ner, train_gpt2, compare
app/                Flask server + one-page front end (templates/, static/)
data/ models/       generated: dataset sample, trained weights (git-ignored)
outputs/            comparison.json / comparison.md deliverable
```
