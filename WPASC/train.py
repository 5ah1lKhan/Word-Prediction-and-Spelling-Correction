# -*- coding: utf-8 -*-
"""Script to train the LSTM text generation model."""

import os
import sys
import argparse

# Add project root to the Python path to allow importing from src
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import (
    load_text_from_file,
    tokenize_text,
    build_vocabulary,
    create_sequences,
    save_preprocessing_data
)
from model import (
    build_lstm_model,
    train_model,
    save_model_and_history
)

# --- Configuration --- #
# Default paths (can be overridden by command-line arguments)
DEFAULT_DATA_PATH = "data/joined_text.txt" # Path to the input text corpus
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_SAVE_PATH = os.path.join(DEFAULT_MODEL_DIR, "lstm_text_gen_model.h5")
HISTORY_SAVE_PATH = os.path.join(DEFAULT_MODEL_DIR, "training_history.pkl")
TOKEN_INDEX_SAVE_PATH = os.path.join(DEFAULT_MODEL_DIR, "token_index.pkl")
METADATA_SAVE_PATH = os.path.join(DEFAULT_MODEL_DIR, "metadata.pkl")

# Model Hyperparameters (can be overridden)
DEFAULT_SEQUENCE_LENGTH = 10 # As per reference notebook
DEFAULT_EPOCHS = 10 # As per reference notebook (initial run)
DEFAULT_BATCH_SIZE = 128 # As per reference notebook
DEFAULT_TEXT_LIMIT = 10000 # Limit text size for faster training (as in notebook)

def main(args):
    """Main training workflow."""
    print("--- Starting Training Process ---")

    # 1. Load Data
    print(f"Loading text data from: {args.data_path}")
    full_text = load_text_from_file(args.data_path)
    if not full_text:
        print("Failed to load data. Exiting.")
        return

    # Limit text size if specified
    if args.text_limit > 0:
        print(f"Limiting text to first {args.text_limit} characters.")
        text_to_process = full_text[:args.text_limit]
    else:
        text_to_process = full_text
    print(f"Processing text length: {len(text_to_process)} characters.")

    # 2. Preprocess Data
    print("Tokenizing text...")
    tokens = tokenize_text(text_to_process)
    if not tokens:
        print("No tokens found after preprocessing. Exiting.")
        return
    print(f"Total tokens: {len(tokens)}")

    print("Building vocabulary...")
    unique_tokens, unique_token_index = build_vocabulary(tokens)
    vocab_size = len(unique_tokens)
    print(f"Vocabulary size: {vocab_size}")

    print("Creating sequences...")
    X, y = create_sequences(tokens, args.sequence_length, unique_token_index)
    if X.size == 0 or y.size == 0:
        print("Failed to create sequences (perhaps text is too short?). Exiting.")
        return
    print(f"Created {X.shape[0]} sequences.")
    print(f"X shape: {X.shape}") # (num_sequences, sequence_length, vocab_size)
    print(f"y shape: {y.shape}") # (num_sequences, vocab_size)

    # 3. Build Model
    print("Building LSTM model...")
    model = build_lstm_model(args.sequence_length, vocab_size)

    # 4. Train Model
    print(f"Training model for {args.epochs} epochs...")
    history = train_model(model, X, y, batch_size=args.batch_size, epochs=args.epochs, shuffle=True)

    # 5. Save Model, History, and Preprocessing Data
    print("Saving model, history, and metadata...")
    save_model_and_history(model, history, args.model_save_path, args.history_save_path)
    save_preprocessing_data(unique_token_index, args.token_index_save_path)
    metadata = {
        "sequence_length": args.sequence_length,
        "vocab_size": vocab_size,
        "unique_tokens": unique_tokens
    }
    save_preprocessing_data(metadata, args.metadata_save_path)

    print("--- Training Process Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for text generation.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help=f"Path to the input text file (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--text_limit", type=int, default=DEFAULT_TEXT_LIMIT,
                        help="Limit the number of characters read from the data file (0 for no limit). Default: 1,000,000")
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help=f"Length of input word sequences (default: {DEFAULT_SEQUENCE_LENGTH})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--model_save_path", type=str, default=DEFAULT_MODEL_SAVE_PATH,
                        help=f"Path to save the trained model (default: {DEFAULT_MODEL_SAVE_PATH})")
    parser.add_argument("--history_save_path", type=str, default=HISTORY_SAVE_PATH,
                        help=f"Path to save the training history (default: {HISTORY_SAVE_PATH})")
    parser.add_argument("--token_index_save_path", type=str, default=TOKEN_INDEX_SAVE_PATH,
                        help=f"Path to save the token index mapping (default: {TOKEN_INDEX_SAVE_PATH})")
    parser.add_argument("--metadata_save_path", type=str, default=METADATA_SAVE_PATH,
                        help=f"Path to save model metadata (vocab, seq_len) (default: {METADATA_SAVE_PATH})")

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.token_index_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_save_path), exist_ok=True)

    main(args)

