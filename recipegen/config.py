from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

RECIPES_JSONL = DATA_DIR / "recipes.jsonl"

GPT2_PRESET = "gpt2_base_en"
GPT2_FINETUNED_DIR = MODELS_DIR / "gpt2_recipes"
TRAINING_REPORT = MODELS_DIR / "gpt2_recipes" / "training_report.json"

BERT_PRESET = "bert_tiny_en_uncased"
NER_DIR = MODELS_DIR / "bert_ner"

DIFFUSION_MODEL = "stabilityai/sd-turbo"

COMPARISON_JSON = OUTPUTS_DIR / "comparison.json"
COMPARISON_MD = OUTPUTS_DIR / "comparison.md"

# The five ingredient sets used for the base vs fine-tuned deliverable.
EVAL_INGREDIENT_SETS = [
    ["chicken thighs", "garlic", "lemon", "rosemary", "potatoes"],
    ["spaghetti", "tomatoes", "basil", "parmesan", "olive oil"],
    ["flour", "butter", "sugar", "eggs", "chocolate chips"],
    ["chickpeas", "coconut milk", "onion", "curry powder", "spinach"],
    ["salmon", "soy sauce", "honey", "ginger", "rice"],
]
