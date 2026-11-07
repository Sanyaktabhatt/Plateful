"""Rule-based parsing of ingredient lines ("1 1/2 c. brown sugar, packed").

Used for three things:
  * deriving the bare ingredient names that go in the prompt (data prep),
  * producing weak QTY / UNIT / ING labels to train the BERT tagger (U3),
  * a fallback tagger when the BERT model has not been trained yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

UNICODE_FRACTIONS = {
    "½": 0.5, "⅓": 1 / 3, "⅔": 2 / 3, "¼": 0.25, "¾": 0.75,
    "⅕": 0.2, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}

# canonical unit -> (kind, factor to ml or g)
UNITS = {
    "tsp": ("vol", 4.93), "tbsp": ("vol", 14.79), "cup": ("vol", 240.0),
    "fl oz": ("vol", 29.57), "pint": ("vol", 473.0), "quart": ("vol", 946.0),
    "gallon": ("vol", 3785.0), "ml": ("vol", 1.0), "l": ("vol", 1000.0),
    "pinch": ("vol", 0.31), "dash": ("vol", 0.62),
    "g": ("mass", 1.0), "kg": ("mass", 1000.0), "oz": ("mass", 28.35),
    "lb": ("mass", 453.6), "stick": ("mass", 113.0),
    # containers and counts: no reliable conversion
    "can": ("pkg", None), "jar": ("pkg", None), "pkg": ("pkg", None),
    "box": ("pkg", None), "bottle": ("pkg", None), "carton": ("pkg", None),
    "envelope": ("pkg", None), "bag": ("pkg", None),
    "clove": ("count", None), "slice": ("count", None), "bunch": ("count", None),
    "head": ("count", None), "sprig": ("count", None), "piece": ("count", None),
    "handful": ("count", None),
}

UNIT_ALIASES = {
    "teaspoon": "tsp", "teaspoons": "tsp", "tsp": "tsp", "tsps": "tsp", "t": "tsp",
    "tablespoon": "tbsp", "tablespoons": "tbsp", "tbsp": "tbsp", "tbsps": "tbsp",
    "tbs": "tbsp", "tbl": "tbsp", "T": "tbsp",
    "cup": "cup", "cups": "cup", "c": "cup",
    "pint": "pint", "pints": "pint", "pt": "pint",
    "quart": "quart", "quarts": "quart", "qt": "quart", "qts": "quart",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "pinch": "pinch", "pinches": "pinch", "dash": "dash", "dashes": "dash",
    "g": "g", "gram": "g", "grams": "g", "gr": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "stick": "stick", "sticks": "stick",
    "can": "can", "cans": "can", "jar": "jar", "jars": "jar",
    "pkg": "pkg", "pkgs": "pkg", "package": "pkg", "packages": "pkg",
    "box": "box", "boxes": "box", "bottle": "bottle", "bottles": "bottle",
    "carton": "carton", "cartons": "carton", "envelope": "envelope",
    "envelopes": "envelope", "bag": "bag", "bags": "bag",
    "clove": "clove", "cloves": "clove", "slice": "slice", "slices": "slice",
    "bunch": "bunch", "bunches": "bunch", "head": "head", "heads": "head",
    "sprig": "sprig", "sprigs": "sprig", "piece": "piece", "pieces": "piece",
    "handful": "handful", "handfuls": "handful",
}

DESCRIPTORS = set("""
chopped minced diced sliced grated shredded crushed fresh freshly large small medium
finely coarsely roughly thinly thickly firmly packed lightly softened melted cold warm
hot room temperature boneless skinless peeled seeded halved quartered cubed beaten
whole dried ground toasted cooked uncooked raw frozen thawed drained rinsed canned
optional divided extra virgin heaping level scant about approximately plus more
to taste for serving garnish good quality best organic sifted unsifted pitted trimmed
""".split())

CONNECTORS = {"of", "and", "&", "with"}

_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|[½⅓⅔¼¾⅕⅛⅜⅝⅞])"
_QTY_RE = re.compile(rf"^\s*(-?\s*{_NUM}(?:\s*[½⅓⅔¼¾⅛])?(?:\s*(?:-|to)\s*{_NUM})?)\s*")


@dataclass
class ParsedIngredient:
    text: str
    quantity: float | None = None
    unit: str | None = None
    name: str = ""
    negative: bool = False
    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def unit_kind(self):
        return UNITS.get(self.unit, (None, None))[0] if self.unit else None

    def amount_metric(self):
        """Return (value, 'ml'|'g') if convertible, else None."""
        if self.quantity is None or not self.unit:
            return None
        kind, factor = UNITS.get(self.unit, (None, None))
        if factor is None:
            return None
        return self.quantity * factor, ("ml" if kind == "vol" else "g")


def _num(tok: str) -> float:
    tok = tok.strip()
    total = 0.0
    for part in tok.split():
        if part in UNICODE_FRACTIONS:
            total += UNICODE_FRACTIONS[part]
        elif "/" in part:
            a, b = part.split("/", 1)
            total += float(a) / float(b) if float(b) else 0.0
        else:
            m = re.match(r"(\d+(?:\.\d+)?)([½⅓⅔¼¾⅛]?)", part)
            if m:
                total += float(m.group(1)) + UNICODE_FRACTIONS.get(m.group(2), 0.0)
    return total


def parse_quantity(s: str) -> float | None:
    s = s.replace("-", " - ") if re.search(r"\d\s*-\s*\d", s) else s
    parts = re.split(r"\s+(?:-|to)\s+", s.strip())
    vals = [_num(p) for p in parts if p.strip() and p.strip() != "-"]
    if not vals:
        return None
    return max(vals)  # for ranges, judge the upper bound


def normalize_unit(word: str) -> str | None:
    w = word.strip(".,()")
    if w in UNIT_ALIASES:  # case-sensitive first, for "T" vs "t"
        return UNIT_ALIASES[w]
    return UNIT_ALIASES.get(w.lower())


def _core_name(rest: str) -> str:
    rest = re.sub(r"\([^)]*\)", " ", rest)
    rest = rest.split(",")[0]
    rest = re.split(r"\s+or\s+|\s+for\s+|\s+to taste", rest)[0]
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z'\-]*", rest.lower())
             if w not in DESCRIPTORS and w not in ("a", "an", "the", "into", "in")]
    words = words[-5:]
    while words and words[0] in CONNECTORS:
        words.pop(0)
    while words and words[-1] in CONNECTORS:
        words.pop()
    return " ".join(words).strip()


def parse_line(line: str) -> ParsedIngredient:
    text = re.sub(r"\s+", " ", line.strip())
    text = re.sub(r"^(?:[•*]\s*|-\s+)", "", text).strip()
    p = ParsedIngredient(text=text)
    rest = text
    m = _QTY_RE.match(rest)
    if m and re.search(r"\d|[½⅓⅔¼¾⅕⅛⅜⅝⅞]", m.group(1)):
        raw_q = m.group(1)
        p.negative = raw_q.strip().startswith("-")
        p.quantity = parse_quantity(raw_q.lstrip("- "))
        if p.negative and p.quantity is not None:
            p.quantity = -p.quantity
        rest = rest[m.end():]
    # skip a parenthetical package size: "2 (16 oz.) pkg. corn"
    pm = re.match(r"^\(([^)]*)\)\s*", rest)
    if pm:
        rest = rest[pm.end():]
    first = rest.split(" ", 1)
    if first and first[0]:
        u = normalize_unit(first[0])
        if u == "tsp" and first[0].strip(".") == "t" and p.quantity is None:
            u = None
        if u:
            p.unit = u
            rest = first[1] if len(first) > 1 else ""
            if rest.lower().startswith("of "):
                rest = rest[3:]
    p.name = _core_name(rest)
    p.tokens, p.labels = weak_labels(text, p)
    return p


def weak_labels(text: str, p: ParsedIngredient):
    """BIO labels over whitespace tokens: QTY, UNIT, ING, everything else O."""
    tokens = text.split()
    labels = ["O"] * len(tokens)
    i = 0
    # quantity tokens at the start
    while i < len(tokens) and re.fullmatch(r"-?[\d/.\-½⅓⅔¼¾⅕⅛⅜⅝⅞]+|to", tokens[i]):
        if tokens[i] == "to" and i == 0:
            break
        labels[i] = "B-QTY" if i == 0 or labels[i - 1] == "O" else "I-QTY"
        i += 1
    # package size in parentheses counts as O; unit word after
    j = i
    if j < len(tokens) and tokens[j].startswith("("):
        while j < len(tokens) and not tokens[j].endswith(")"):
            j += 1
        j += 1
    if p.unit and j < len(tokens) and normalize_unit(tokens[j]) == p.unit:
        labels[j] = "B-UNIT"
    # ingredient name span: match the core-name words in order
    if p.name:
        name_words = p.name.split()
        low = [re.sub(r"[^a-z'\-]", "", t.lower()) for t in tokens]
        for start in range(len(tokens)):
            if low[start] == name_words[0]:
                k, matched = start, 0
                while k < len(tokens) and matched < len(name_words):
                    if low[k] == name_words[matched]:
                        labels[k] = "B-ING" if matched == 0 else "I-ING"
                        matched += 1
                    elif labels[k] == "O" and low[k] in DESCRIPTORS:
                        pass
                    else:
                        break
                    k += 1
                if matched == len(name_words):
                    break
                for x in range(start, k):  # undo partial match
                    if labels[x].endswith("ING"):
                        labels[x] = "O"
    return tokens, labels


def split_user_input(text: str) -> list[str]:
    """Split what a user typed (commas, newlines, semicolons) into lines."""
    parts = re.split(r"[\n;,]+|\s+\band\b\s+(?=[\d½¼¾a-zA-Z])", text)
    return [p.strip(" -•*\t") for p in parts if p.strip(" -•*\t")]
