"""Larder: the recipe app. Flask backend serving one page plus a small JSON API.

    python app/server.py            # http://127.0.0.1:5045
"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recipegen  # noqa: E402,F401
from flask import Flask, jsonify, render_template, request  # noqa: E402
from recipegen import config, imagegen, model, ner, pipeline  # noqa: E402
from recipegen.safety import ALLERGENS  # noqa: E402

app = Flask(__name__)
# Keras generation and the diffusion pipeline are not safe to run concurrently.
gen_lock = threading.Lock()
img_lock = threading.Lock()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


@app.get("/")
def index():
    return render_template(
        "index.html",
        allergens=list(ALLERGENS),
        sets=config.EVAL_INGREDIENT_SETS,
    )


@app.get("/api/status")
def status():
    report = _read_json(config.TRAINING_REPORT)
    ner_meta = _read_json(config.NER_DIR / "meta.json")
    return jsonify({
        "finetuned": model.finetuned_available(),
        "ner": "bert" if ner.available() else "rules",
        "image": imagegen.available(),
        "training": report and {k: report[k] for k in (
            "train_recipes", "val_recipes", "epochs", "steps", "batch_size", "sequence_length",
            "learning_rate", "train_minutes", "base", "finetuned", "framework", "loss_curve") if k in report},
        "ner_meta": ner_meta and {k: ner_meta.get(k) for k in ("preset", "train_lines", "val_entity_metrics")},
    })


@app.post("/api/generate")
def generate():
    body = request.get_json(force=True)
    text = (body.get("ingredients") or "").strip()
    which = body.get("model", "finetuned")
    if which not in ("base", "finetuned", "both"):
        return jsonify(error="model must be base, finetuned or both"), 400
    if not text:
        return jsonify(error="Add at least two ingredients."), 400
    if which in ("finetuned", "both") and not model.finetuned_available():
        return jsonify(error="The fine-tuned model hasn't been trained yet. Run scripts/train_gpt2.py, "
                             "or choose the base model."), 409
    try:
        with gen_lock:
            # No text seed: re-seeding recompiles the XLA sampler (~10 s). The sampler's
            # own RNG state already gives a fresh recipe on every call.
            result = pipeline.run(text, which=which, seed=None, avoid=body.get("avoid") or [])
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(result)


@app.post("/api/image")
def image():
    body = request.get_json(force=True)
    with img_lock:
        out = imagegen.generate_image(body.get("title", ""), body.get("ingredients", []),
                                      seed=int(body.get("seed") or 0))
    return jsonify(out)


@app.post("/api/tag")
def tag():
    tagged, engine = ner.tag_user_input(request.get_json(force=True).get("ingredients", ""))
    return jsonify(tagged=tagged, engine=engine)


@app.get("/api/comparison")
def comparison():
    data = _read_json(config.COMPARISON_JSON)
    if data is None:
        return jsonify(error="No comparison yet. Run scripts/compare.py."), 404
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5045, debug=False, threaded=True)
