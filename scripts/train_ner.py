"""U3 + U4: train the BERT ingredient tagger in TensorFlow 2 (Keras 3, TF backend)."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recipegen  # noqa: E402,F401
from recipegen import config  # noqa: E402
from recipegen.data import load_recipes  # noqa: E402
from recipegen.ingredients import parse_line  # noqa: E402
from recipegen.ner import LABELS, build_model, encode, get_tokenizer, spans_from_labels  # noqa: E402


def augment(tokens, labels, rng):
    """Make the model robust to how people type: lowercase, no quantity, no unit."""
    r = rng.random()
    if r < 0.15:  # drop the quantity + unit, like "brown sugar"
        keep = [i for i, l in enumerate(labels) if not l.endswith(("QTY", "UNIT"))]
        return [tokens[i] for i in keep], [labels[i] for i in keep]
    if r < 0.3:
        return [t.lower() for t in tokens], labels
    return tokens, labels


def entity_f1(true_lines, pred_lines, tag="ING"):
    tp = fp = fn = 0
    for t, p in zip(true_lines, pred_lines):
        ts = {s["text"] for s in t if s["tag"] == tag}
        ps = {s["text"] for s in p if s["tag"] == tag}
        tp += len(ts & ps); fp += len(ps - ts); fn += len(ts - ps)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--preset", default=config.BERT_PRESET)
    args = ap.parse_args()

    import keras
    import numpy as np
    import tensorflow as tf

    keras.utils.set_random_seed(3)
    rng = random.Random(3)
    lines = [l for r in load_recipes() for l in r["ingredient_lines"]]
    rng.shuffle(lines)
    lines = lines[: args.lines]
    parsed = [parse_line(l) for l in lines]
    data = [augment(p.tokens, p.labels, rng) for p in parsed if p.tokens]
    data = [d for d in data if d[0]]
    split = int(len(data) * 0.9)
    train, val = data[:split], data[split:]
    print(f"{len(train)} train / {len(val)} val ingredient lines")

    tok = get_tokenizer(args.preset)
    xtr, ytr, wtr, _ = encode(tok, [d[0] for d in train], [d[1] for d in train])
    xva, yva, wva, wi_va = encode(tok, [d[0] for d in val], [d[1] for d in val])

    train_ds = tf.data.Dataset.from_tensor_slices((xtr, ytr, wtr)).shuffle(len(train), seed=3) \
        .batch(args.batch).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((xva, yva, wva)).batch(args.batch)

    model = build_model(args.preset)
    model.compile(
        optimizer=keras.optimizers.AdamW(3e-4, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=["accuracy"],
    )
    model.summary()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    pred = np.argmax(model.predict(xva, verbose=0), axis=-1)
    pred_spans, true_spans = [], []
    for i, (words, labels) in enumerate(val):
        pl = [LABELS[pred[i, p]] if p is not None else "O" for p in wi_va[i]]
        pred_spans.append(spans_from_labels(words, pl))
        true_spans.append(spans_from_labels(words, labels))
    metrics = {t: entity_f1(true_spans, pred_spans, t) for t in ("ING", "QTY", "UNIT")}
    print(json.dumps(metrics, indent=2))

    config.NER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(config.NER_DIR / "ner.weights.h5"))
    (config.NER_DIR / "meta.json").write_text(json.dumps({
        "preset": args.preset, "labels": LABELS, "train_lines": len(train), "val_lines": len(val),
        "epochs": args.epochs, "val_entity_metrics": metrics,
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
    }, indent=2))
    print("saved to", config.NER_DIR)


if __name__ == "__main__":
    main()
