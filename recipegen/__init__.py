"""Recipe generator: GPT-2 (KerasHub) fine-tuned on RecipeNLG, with BERT NER,
diffusion images and allergen / quantity safety checks."""
import os

# Everything Keras in this project runs on the TensorFlow 2 backend (U4).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
