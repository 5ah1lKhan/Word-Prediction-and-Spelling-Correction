# -*- coding: utf-8 -*-
"""Script to generate text using a pre-trained LSTM model."""

import os
import sys
import argparse
import numpy as np

# Add project root to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import load_preprocessing_data, tokenize_text
from model import load_trained_model
from generator import generate_text

# --- Configuration --- #
# Default paths (should match the paths used in train.py)
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "lstm_text_gen_model.h5")
DEFAULT_TOKEN_INDEX_PATH = os.path.join(DEFAULT_MODEL_DIR, "token_index.pkl")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "metadata.pkl")

# Generation parameters
DEFAULT_NUM_WORDS = 100 # Number of words to generate
DEFAULT_CREATIVITY = 5  # Number of top predictions to consider (higher = more random)

def main(args):
    """Main text generation workflow."""
    print("--- Starting Text Generation Process ---")

    # 1. Load Model and Preprocessing Data
    print(f"Loading model from: {args.model_path}")
    model = load_trained_model(args.model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    print(f"Loading token index from: {args.token_index_path}")
    token_index = load_preprocessing_data(args.token_index_path)
    if token_index is None:
        print("Failed to load token index. Exiting.")
        return

    print(f"Loading metadata from: {args.metadata_path}")
    metadata = load_preprocessing_data(args.metadata_path)
    if metadata is None:
        print("Failed to load metadata. Exiting.")
        return

    sequence_length = metadata.get("sequence_length")
    unique_tokens = metadata.get("unique_tokens")
    vocab_size = metadata.get("vocab_size")

    if not all([sequence_length, unique_tokens, vocab_size]):
        print("Error: Metadata file is missing required keys (sequence_length, unique_tokens, vocab_size). Exiting.")
        return

    # 2. Prepare Seed Text
    seed_text = args.seed_text
    print(f"\nUsing seed text: \"{seed_text}\"")

    # Basic validation of seed text length
    seed_tokens = tokenize_text(seed_text)
    if len(seed_tokens) < sequence_length:
        print(f"\nError: Seed text must contain at least {sequence_length} words after tokenization.")
        print(f"Provided seed text has {len(seed_tokens)} words: {seed_tokens}")
        print("Please provide a longer seed text.")
        return

    # 3. Generate Text
    print(f"Generating {args.num_words} words with creativity level {args.creativity}...")
    generated_output = generate_text(
        model=model,
        token_index=token_index,
        unique_tokens=unique_tokens,
        seed_text=seed_text,
        sequence_length=sequence_length,
        num_words_to_generate=args.num_words,
        creativity=args.creativity
    )

    # 4. Print Result
    print("\n--- Generated Text ---")
    print(generated_output)
    print("\n--- Generation Process Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained LSTM model.")
    parser.add_argument("seed_text", type=str,
                        help="The starting text sequence (must be at least sequence_length words long).")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the trained model (.h5 file) (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--token_index_path", type=str, default=DEFAULT_TOKEN_INDEX_PATH,
                        help=f"Path to the saved token index mapping (default: {DEFAULT_TOKEN_INDEX_PATH})")
    parser.add_argument("--metadata_path", type=str, default=DEFAULT_METADATA_PATH,
                        help=f"Path to the saved model metadata (default: {DEFAULT_METADATA_PATH})")
    parser.add_argument("--num_words", type=int, default=DEFAULT_NUM_WORDS,
                        help=f"Number of words to generate after the seed text (default: {DEFAULT_NUM_WORDS})")
    parser.add_argument("--creativity", type=int, default=DEFAULT_CREATIVITY,
                        help=f"Number of top predictions to sample from (higher is more random, default: {DEFAULT_CREATIVITY})")

    args = parser.parse_args()

    main(args)

