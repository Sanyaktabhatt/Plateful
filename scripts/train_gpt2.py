"""U2 + U4: fine-tune GPT-2 (KerasHub) on recipes, training in TensorFlow 2.

Keras 3 runs on the TensorFlow backend here: data goes through tf.data, and
fit() runs as compiled tf.functions. The base model's validation loss is
measured before training, so the fine-tuned model has a number to beat.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recipegen  # noqa: E402,F401  (sets KERAS_BACKEND)
from recipegen import config  # noqa: E402
from recipegen.data import load_recipes, training_texts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--no-xla", dest="xla", action="store_false")
    ap.add_argument("--out", default=str(config.GPT2_FINETUNED_DIR))
    args = ap.parse_args()

    import keras
    import keras_hub
    import tensorflow as tf

    print("TensorFlow", tf.__version__, "| Keras", keras.__version__, "| backend", keras.backend.backend())
    keras.utils.set_random_seed(7)

    preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
        config.GPT2_PRESET, sequence_length=args.seq_len)

    # Keep only recipes that fit whole (start token + text), so every example
    # ends in <|end|> and the model learns to stop.
    texts, n_long = [], 0
    for t in training_texts(load_recipes()):
        if len(preprocessor.tokenizer(t)) <= args.seq_len - 1:
            texts.append(t)
        else:
            n_long += 1
        if len(texts) >= args.samples + args.val:
            break
    train_texts, val_texts = texts[args.val:], texts[: args.val]
    print(f"{len(train_texts)} train / {len(val_texts)} val recipes ({n_long} skipped as too long)")
    lm = keras_hub.models.GPT2CausalLM.from_preset(config.GPT2_PRESET, preprocessor=preprocessor)

    def make_ds(t, shuffle):
        ds = tf.data.Dataset.from_tensor_slices(t)
        if shuffle:
            ds = ds.shuffle(len(t), seed=7, reshuffle_each_iteration=True)
        return ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_ds(train_texts, True), make_ds(val_texts, False)

    steps = math.ceil(len(train_texts) / args.batch) * args.epochs
    schedule = keras.optimizers.schedules.PolynomialDecay(args.lr, steps, end_learning_rate=args.lr * 0.1)
    lm.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=["accuracy"],
        jit_compile=args.xla,  # XLA: fuses the transformer ops, much faster on CPU
    )

    print("evaluating base GPT-2 on validation recipes...")
    base_loss, base_acc = lm.evaluate(val_ds, verbose=0)
    print(f"base: loss {base_loss:.3f}  perplexity {math.exp(base_loss):.1f}")

    class StepLog(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.points, self.step = [], 0

        def on_train_batch_end(self, batch, logs=None):
            self.step += 1
            if self.step % 10 == 0 and logs:
                self.points.append([self.step, float(logs["loss"])])

    step_log = StepLog()
    t0 = time.time()
    hist = lm.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[step_log])
    train_min = (time.time() - t0) / 60
    ft_loss, ft_acc = lm.evaluate(val_ds, verbose=0)
    print(f"fine-tuned: loss {ft_loss:.3f}  perplexity {math.exp(ft_loss):.1f}  ({train_min:.1f} min)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm.save_to_preset(str(out))
    report = {
        "framework": {"tensorflow": tf.__version__, "keras": keras.__version__,
                      "keras_hub": keras_hub.__version__, "backend": keras.backend.backend()},
        "preset": config.GPT2_PRESET,
        "train_recipes": len(train_texts), "val_recipes": len(val_texts),
        "epochs": args.epochs, "batch_size": args.batch, "sequence_length": args.seq_len,
        "learning_rate": args.lr, "steps": steps, "train_minutes": round(train_min, 1),
        "base": {"val_loss": base_loss, "val_perplexity": math.exp(base_loss), "val_token_accuracy": base_acc},
        "finetuned": {"val_loss": ft_loss, "val_perplexity": math.exp(ft_loss), "val_token_accuracy": ft_acc},
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        "loss_curve": step_log.points,
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
