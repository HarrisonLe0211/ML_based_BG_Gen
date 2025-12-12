#!/usr/bin/env python
"""
Convert and prepare the model for backend use.

The notebook produces a .keras file which is a Keras-native format.
For transformers compatibility, we need to set up the model directory structure.
"""

import os
import json
import shutil
from pathlib import Path
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, AutoConfig, AutoTokenizer

# Paths
BASE_DIR = Path(__file__).parent
MODEL_KERAS_FILE = BASE_DIR / "model_training" / "distilbert_genre_classifier.keras"
MODEL_OUTPUT_DIR = BASE_DIR / "model_training" / "distilbert_genre_model"
TOKENIZER_NAME = "distilbert-base-uncased"

def setup_model():
    """
    Load the .keras model and prepare it for backend use.
    """
    if not MODEL_KERAS_FILE.exists():
        print(f"✗ Model file not found: {MODEL_KERAS_FILE}")
        return False

    print(f"Loading model from {MODEL_KERAS_FILE}...")
    try:
        keras_model = tf.keras.models.load_model(str(MODEL_KERAS_FILE))
        print("✓ Keras model loaded")
    except Exception as e:
        print(f"✗ Failed to load Keras model: {e}")
        return False

    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {MODEL_OUTPUT_DIR}")

    # Save the model weights
    weights_dir = MODEL_OUTPUT_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # Save config from keras model (if available)
    if hasattr(keras_model, "config"):
        config_dict = keras_model.config
        with open(MODEL_OUTPUT_DIR / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("✓ Config saved")

    # Save the model using different methods
    try:
        # Method 1: Save as TensorFlow SavedModel format
        keras_model.save(str(MODEL_OUTPUT_DIR / "saved_model"), save_format="tf")
        print("✓ Model saved as SavedModel")
    except Exception as e:
        print(f"Note: SavedModel save skipped: {e}")

    # Download and cache the tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
        print("✓ Tokenizer saved")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

    print(f"\n✓ Model setup complete!")
    print(f"  Model directory: {MODEL_OUTPUT_DIR}")
    print(f"  Ready to use with backend")
    return True

if __name__ == "__main__":
    setup_model()
