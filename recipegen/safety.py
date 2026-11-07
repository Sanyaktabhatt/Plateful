"""U5: allergen warnings and impossible-quantity flags.

Runs on both the user's ingredient list and the generated recipe, since a
generated recipe can add allergens (butter, soy sauce) the user never typed.
"""
from __future__ import annotations

import re

from .ingredients import ParsedIngredient, parse_line

# Major allergens: US "Big 9" plus the extra EU-14 ones that come up in cooking.
ALLERGENS = {
    "Milk": ["milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "ghee", "whey",
             "casein", "buttermilk", "parmesan", "mozzarella", "cheddar", "ricotta",
             "mascarpone", "feta", "paneer", "custard", "half-and-half", "ice cream",
             "sour cream", "creme fraiche", "brie", "gouda", "romano", "kefir"],
    "Egg": ["egg", "eggs", "yolk", "yolks", "mayonnaise", "mayo", "meringue", "aioli"],
    "Peanut": ["peanut", "peanuts", "groundnut"],
    "Tree nuts": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
                  "macadamia", "pine nut", "brazil nut", "praline", "marzipan",
                  "nutella", "nuts"],
    "Wheat / gluten": ["flour", "wheat", "bread", "breadcrumbs", "pasta", "spaghetti",
                       "noodle", "noodles", "macaroni", "couscous", "barley", "rye",
                       "semolina", "bulgur", "farro", "spelt", "tortilla", "pita",
                       "cracker", "crackers", "biscuit", "biscuits", "soy sauce",
                       "panko", "seitan", "orzo", "lasagna", "penne", "fettuccine",
                       "cake mix", "pie crust", "croutons", "beer"],
    "Soy": ["soy", "soya", "tofu", "edamame", "miso", "tempeh", "soy sauce", "tamari"],
    "Fish": ["fish", "salmon", "tuna", "cod", "anchovy", "anchovies", "sardine",
             "tilapia", "halibut", "trout", "haddock", "mackerel", "fish sauce",
             "worcestershire", "snapper", "catfish", "bass"],
    "Shellfish": ["shrimp", "prawn", "prawns", "crab", "lobster", "clam", "clams",
                  "mussel", "mussels", "oyster", "oysters", "scallop", "scallops",
                  "crawfish", "crayfish", "langoustine", "squid", "calamari", "octopus"],
    "Sesame": ["sesame", "tahini", "benne"],
    "Mustard": ["mustard"],
    "Celery": ["celery", "celeriac"],
    "Sulphites": ["wine", "sherry", "dried apricots", "vermouth"],
}

# Phrases that contain an allergen keyword but are not that allergen.
EXCEPTIONS = {
    "Milk": ["coconut milk", "almond milk", "oat milk", "soy milk", "rice milk",
             "cashew milk", "coconut cream", "cream of tartar", "peanut butter",
             "almond butter", "apple butter", "cocoa butter", "butternut",
             "butter beans", "shea butter", "cashew butter", "sunflower butter",
             "dairy-free", "vegan"],
    "Egg": ["eggplant", "egg-free", "eggless", "egg noodles"],
    "Tree nuts": ["nutmeg", "butternut", "coconut", "water chestnut", "doughnut",
                  "donut", "chestnut"],
    "Wheat / gluten": ["buckwheat", "gluten-free", "rice flour", "almond flour",
                       "coconut flour", "corn tortilla", "rice noodles", "cornflour",
                       "corn flour", "chickpea flour", "tapioca flour", "oat flour"],
    "Fish": ["fish sauce-free", "shellfish"],
}

PROTEINS = ["chicken", "beef", "pork", "lamb", "turkey", "salmon", "steak", "breast",
            "thigh", "fillet", "roast", "ham", "sausage", "duck", "shrimp"]

# (keyword list, unusual above, impossible above) in grams or ml.
CAPS = [
    (["salt"], 40, 250),
    (["baking soda", "baking powder", "bicarbonate"], 25, 150),
    (["yeast"], 30, 200),
    (["vanilla extract", "almond extract", "vanilla", "extract"], 45, 250),
    (["cinnamon", "cumin", "paprika", "turmeric", "nutmeg", "cayenne", "chili powder",
      "curry powder", "oregano", "thyme", "garam masala", "cloves", "pepper flakes",
      "black pepper", "ginger powder", "allspice", "coriander"], 35, 250),
    (["saffron"], 2, 20),
]
ANY_UNUSUAL, ANY_IMPOSSIBLE = 4000, 20000  # grams / ml for a single ingredient
COUNT_UNUSUAL, COUNT_IMPOSSIBLE = 30, 200


def _has(text: str, kw: str) -> bool:
    # allow simple plurals: almond -> almonds, peach -> peaches
    return re.search(rf"(?<![a-z]){re.escape(kw)}(?:e?s)?(?![a-z])", text) is not None


def _strip_exceptions(text: str, allergen: str) -> str:
    # Also strip other allergens' exceptions that hide a false keyword, e.g.
    # "pine nut" is a tree nut but "nut" alone would double-count.
    for phrase in EXCEPTIONS.get(allergen, []):
        text = text.replace(phrase, " ")
    return text


def detect_allergens(lines: list[str]) -> dict[str, list[str]]:
    """Return {allergen: [ingredient lines that trigger it]}."""
    found: dict[str, list[str]] = {}
    for line in lines:
        low = " " + line.lower() + " "
        for allergen, kws in ALLERGENS.items():
            text = _strip_exceptions(low, allergen)
            # Explicit allergen-free variants: don't flag.
            if allergen == "Milk" and re.search(r"dairy[- ]free|vegan", low):
                continue
            hit = next((kw for kw in kws if _has(text, kw)), None)
            if hit:
                found.setdefault(allergen, [])
                if line not in found[allergen]:
                    found[allergen].append(line)
    return found


def _fmt(v, u):
    if u == "g" and v >= 1000:
        return f"{v / 1000:.1f} kg"
    if u == "ml" and v >= 1000:
        return f"{v / 1000:.1f} L"
    return f"{v:.0f} {u}"


def check_quantity(p: ParsedIngredient) -> dict | None:
    """Flag one parsed ingredient line. Returns None if it looks fine."""
    def flag(level, reason):
        return {"line": p.text, "level": level, "reason": reason}

    if p.quantity is None:
        return None
    if p.quantity < 0:
        return flag("impossible", "Negative quantity.")
    if p.quantity == 0:
        return flag("impossible", "Zero quantity: the ingredient is listed but none is used.")

    name = (p.name or p.text).lower()
    metric = p.amount_metric()

    if p.unit in ("tsp", "pinch", "dash") and any(_has(name, w) for w in PROTEINS):
        return flag("unusual", f"Measuring {p.name} by the {p.unit} doesn't make sense for a whole protein.")

    if metric:
        value, unit = metric
        for kws, unusual, impossible in CAPS:
            if any(_has(name, k) for k in kws):
                if value > impossible:
                    return flag("impossible", f"{_fmt(value, unit)} of {p.name} would make the dish inedible.")
                if value > unusual:
                    return flag("unusual", f"{_fmt(value, unit)} of {p.name} is far more than a home recipe uses.")
                return None
        if value > ANY_IMPOSSIBLE:
            return flag("impossible", f"{_fmt(value, unit)} of a single ingredient is not a home-kitchen amount.")
        if value > ANY_UNUSUAL:
            return flag("unusual", f"{_fmt(value, unit)} of {p.name} is a very large amount.")
        return None

    if p.unit is None or p.unit_kind == "count":
        if p.quantity > COUNT_IMPOSSIBLE:
            return flag("impossible", f"{p.quantity:g} {p.name} is not a realistic count.")
        limit = 18 if "egg" in name else COUNT_UNUSUAL
        if p.quantity > limit:
            return flag("unusual", f"{p.quantity:g} {p.name} is a lot for one recipe.")
        if "egg" in name and 0 < p.quantity < 0.25:
            return flag("unusual", "Less than a quarter of an egg can't really be measured.")
    if p.unit_kind == "pkg" and p.quantity > 20:
        return flag("unusual", f"{p.quantity:g} {p.unit}s of {p.name} is a lot for one recipe.")
    return None


_TEMP_RE = re.compile(r"(\d{2,4})\s*(?:°|degrees?|deg\.?)\s*([FC])?", re.I)
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
_MIN_RE = re.compile(r"(\d+)\s*(?:minutes?|mins?)\b", re.I)


def check_directions(steps: list[str]) -> list[dict]:
    flags = []
    for step in steps:
        low = step.lower()
        for m in _TEMP_RE.finditer(step):
            t = int(m.group(1))
            scale = (m.group(2) or "").upper()
            celsius = scale == "C" or (not scale and t <= 260 and "°c" in low)
            if celsius:
                if t > 300:
                    flags.append({"line": step, "level": "impossible", "reason": f"{t}°C is beyond any home oven."})
            elif t > 600:
                flags.append({"line": step, "level": "impossible", "reason": f"{t}°F is beyond any home oven."})
            elif t > 525 and ("oven" in low or "bake" in low):
                flags.append({"line": step, "level": "unusual", "reason": f"{t}°F is hotter than most home ovens go."})
        long_ok = any(w in low for w in ("marinate", "refrigerat", "chill", "slow cooker",
                                          "crock", "overnight", "ferment", "freeze", "soak", "cure"))
        for m in _HOURS_RE.finditer(step):
            h = float(m.group(1))
            if h > (96 if long_ok else 48):
                flags.append({"line": step, "level": "impossible", "reason": f"{h:g} hours is not a realistic cooking time."})
            elif h > 12 and not long_ok:
                flags.append({"line": step, "level": "unusual", "reason": f"{h:g} hours of active cooking is unusually long."})
        for m in _MIN_RE.finditer(step):
            mins = int(m.group(1))
            if mins > 720 and not long_ok:
                flags.append({"line": step, "level": "unusual", "reason": f"{mins} minutes ({mins / 60:.0f} h) is unusually long."})
    return flags


def analyze(ingredient_lines: list[str], directions: list[str] | None = None) -> dict:
    parsed = [parse_line(l) for l in ingredient_lines]
    qty_flags = [f for f in (check_quantity(p) for p in parsed) if f]
    qty_flags += check_directions(directions or [])
    return {
        "allergens": detect_allergens(ingredient_lines),
        "quantity_flags": qty_flags,
    }
