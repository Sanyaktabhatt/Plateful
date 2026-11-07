"""U3 (part 1): BERT token classifier that tags ingredient text.

Labels: QTY (2, 1 1/2), UNIT (cups, tbsp), ING (brown sugar), O (everything else).

There is no labelled ingredient-NER corpus in RecipeNLG, so training labels are
produced by the rule parser in ingredients.py (weak supervision). The BERT
model then generalises past the rules: unseen units, misspellings, free text
such as "a handful of basil", and lines with no quantity at all.
"""
from __future__ import annotations

import json
import threading

from . import config
from .ingredients import normalize_unit, parse_line, split_user_input

LABELS = ["O", "B-QTY", "I-QTY", "B-UNIT", "I-UNIT", "B-ING", "I-ING"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 48

_model = None
_tok = None
_lock = threading.Lock()


def available() -> bool:
    return (config.NER_DIR / "ner.weights.h5").exists()


def get_tokenizer(preset=config.BERT_PRESET):
    import keras_hub
    return keras_hub.tokenizers.BertTokenizer.from_preset(preset)


def build_model(preset=config.BERT_PRESET, load_weights=True):
    import keras
    import keras_hub

    backbone = keras_hub.models.BertBackbone.from_preset(preset, load_weights=load_weights)
    inputs = {
        "token_ids": keras.Input((None,), dtype="int32", name="token_ids"),
        "segment_ids": keras.Input((None,), dtype="int32", name="segment_ids"),
        "padding_mask": keras.Input((None,), dtype="int32", name="padding_mask"),
    }
    seq = backbone(inputs)["sequence_output"]
    seq = keras.layers.Dropout(0.1)(seq)
    logits = keras.layers.Dense(len(LABELS), name="tag_logits")(seq)
    return keras.Model(inputs, logits, name="bert_ingredient_tagger")


def encode(tok, lines_tokens, lines_labels=None):
    """Word-level tokens -> padded WordPiece arrays with aligned labels.

    Each word is tokenised on its own, so we know which word pieces belong to
    which word. The first piece keeps the word's label; continuation pieces of
    a B- word become I-.
    """
    import numpy as np

    flat = [w for ws in lines_tokens for w in ws]
    pieces = []
    if flat:
        out = tok(flat)  # ragged tensor in older KerasHub, list in newer
        out = out.to_list() if hasattr(out, "to_list") else out
        pieces = [[int(x) for x in p] for p in out]
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    n = len(lines_tokens)
    ids = np.full((n, MAX_LEN), pad, dtype="int32")
    mask = np.zeros((n, MAX_LEN), dtype="int32")
    labels = np.zeros((n, MAX_LEN), dtype="int32")
    weights = np.zeros((n, MAX_LEN), dtype="float32")
    word_index = []  # for each line: position of the first piece of each word

    k = 0
    for i, words in enumerate(lines_tokens):
        row, lab, wt, firsts = [cls], [0], [0.0], []
        for j, _ in enumerate(words):
            wp = pieces[k] or [tok.token_to_id("[UNK]")]
            k += 1
            if len(row) + len(wp) >= MAX_LEN - 1:
                firsts.append(None)
                continue
            firsts.append(len(row))
            word_label = lines_labels[i][j] if lines_labels else "O"
            for x, pid in enumerate(wp):
                row.append(pid)
                lbl = word_label
                if x > 0 and lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                lab.append(L2I[lbl])
                wt.append(1.0)
        row.append(sep); lab.append(0); wt.append(0.0)
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
        labels[i, :len(lab)] = lab
        weights[i, :len(wt)] = wt
        word_index.append(firsts)
    feats = {"token_ids": ids, "segment_ids": np.zeros_like(ids), "padding_mask": mask}
    return feats, labels, weights, word_index


def _load():
    global _model, _tok
    with _lock:
        if _model is None:
            meta = json.loads((config.NER_DIR / "meta.json").read_text())
            _tok = get_tokenizer(meta["preset"])
            _model = build_model(meta["preset"], load_weights=False)
            _model.load_weights(str(config.NER_DIR / "ner.weights.h5"))
    return _model, _tok


def spans_from_labels(words, labels):
    out, cur = [], None
    for w, l in zip(words, labels):
        if l == "O":
            cur = None
            out.append({"text": w, "tag": "O"})
            continue
        typ = l[2:]
        if l.startswith("B-") or cur is None or cur["tag"] != typ:
            cur = {"text": w, "tag": typ}
            out.append(cur)
        else:
            cur["text"] += " " + w
    return out


def tag_lines(lines: list[str]) -> tuple[list[dict], str]:
    """Tag each line. Returns ([{line, spans, name, qty, unit}], engine)."""
    lines = [l for l in lines if l.strip()]
    if not lines:
        return [], "none"
    words = [l.split() for l in lines]
    engine = "rules"
    if available():
        import numpy as np
        model, tok = _load()
        feats, _, _, word_index = encode(tok, words)
        pred = np.argmax(model.predict(feats, verbose=0), axis=-1)
        labels = [[LABELS[pred[i, p]] if p is not None else "O" for p in word_index[i]]
                  for i in range(len(lines))]
        engine = "bert"
    else:
        labels = [parse_line(l).labels for l in lines]

    results = []
    for line, ws, ls in zip(lines, words, labels):
        spans = spans_from_labels(ws, ls)
        ing = " ".join(s["text"] for s in spans if s["tag"] == "ING")
        ing = ing.strip(" ,.;:()").lower()
        # "handful of basil" -> "basil": a unit word the tagger folded into the name
        first, _, rest = ing.partition(" ")
        if rest and normalize_unit(first):
            ing = rest[3:] if rest.startswith("of ") else rest
        results.append({
            "line": line,
            "spans": spans,
            "name": ing or parse_line(line).name or line.lower(),
            "qty": next((s["text"] for s in spans if s["tag"] == "QTY"), None),
            "unit": next((s["text"] for s in spans if s["tag"] == "UNIT"), None),
        })
    return results, engine


def tag_user_input(text: str):
    return tag_lines(split_user_input(text))
