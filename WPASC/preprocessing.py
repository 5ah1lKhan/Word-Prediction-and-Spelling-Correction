# -*- coding: utf-8 -*-
"""Data Preprocessing and Tokenization for LSTM Text Generation"""

import re
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
import os

def load_text_from_file(filepath):
    """Loads text data from a file.

    Args:
        filepath (str): Path to the text file.

    Returns:
        str: The content of the file as a single string, or None if error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def tokenize_text(text):
    """Tokenizes text into words using RegexpTokenizer, converting to lowercase.

    Args:
        text (str): The input text string.

    Returns:
        list: A list of tokens (words).
    """
    if not text:
        return []
    # Tokenizer that matches sequences of alphanumeric characters (words)
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text.lower())
    return tokens

def build_vocabulary(tokens):
    """Builds a vocabulary of unique tokens and maps them to indices.

    Args:
        tokens (list): A list of tokens.

    Returns:
        tuple: A tuple containing:
            - list: Sorted list of unique tokens.
            - dict: A dictionary mapping tokens to their indices.
    """
    unique_tokens = sorted(list(set(tokens)))
    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
    return unique_tokens, unique_token_index

def create_sequences(tokens, n_words, unique_token_index):
    """Creates input sequences and corresponding next words.

    Args:
        tokens (list): The list of tokens from the text.
        n_words (int): The number of words in each input sequence.
        unique_token_index (dict): Dictionary mapping tokens to indices.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Input sequences vectorized (samples, n_words, vocab_size).
            - np.ndarray: Target words vectorized (samples, vocab_size).
    """
    input_words_list = []
    next_word_list = []

    if len(tokens) <= n_words:
        print(f"Warning: Text length ({len(tokens)}) is not greater than sequence length ({n_words}). Cannot create sequences.")
        return np.array([]), np.array([])

    for i in range(len(tokens) - n_words):
        input_words_list.append(tokens[i:i + n_words])
        next_word_list.append(tokens[i + n_words])

    num_sequences = len(input_words_list)
    vocab_size = len(unique_token_index)

    # Initialize numpy arrays with zeros
    X = np.zeros((num_sequences, n_words, vocab_size), dtype=bool)
    y = np.zeros((num_sequences, vocab_size), dtype=bool)

    # Populate the arrays
    for i, sequence in enumerate(input_words_list):
        for j, word in enumerate(sequence):
            if word in unique_token_index:
                X[i, j, unique_token_index[word]] = 1
        if next_word_list[i] in unique_token_index:
            y[i, unique_token_index[next_word_list[i]]] = 1

    return X, y

def save_preprocessing_data(data, filepath):
    """Saves preprocessing data (like token index) using pickle.

    Args:
        data: The data to save (e.g., dictionary, list).
        filepath (str): The path to save the pickle file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Preprocessing data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

def load_preprocessing_data(filepath):
    """Loads preprocessing data from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        The loaded data, or None if an error occurs.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Preprocessing data loaded from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: Preprocessing data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

# Example of how to use the functions (can be run standalone for testing)
if __name__ == "__main__":
    # Create dummy data file
    dummy_data_dir = "/home/ubuntu/lstm_text_generation/data"
    dummy_data_path = os.path.join(dummy_data_dir, "sample_text.txt")
    os.makedirs(dummy_data_dir, exist_ok=True)
    sample_content = """
    This is a sample text for testing the preprocessing module.
    It contains several sentences and words.
    Preprocessing involves tokenization and creating sequences.
    """
    with open(dummy_data_path, "w", encoding="utf-8") as f:
        f.write(sample_content)

    print(f"Loading text from: {dummy_data_path}")
    text_content = load_text_from_file(dummy_data_path)

    if text_content:
        print("\nTokenizing text...")
        tokens = tokenize_text(text_content)
        print(f"Tokens: {tokens}")

        print("\nBuilding vocabulary...")
        unique_tokens, unique_token_index = build_vocabulary(tokens)
        print(f"Vocabulary size: {len(unique_tokens)}")
        # print(f"Token index: {unique_token_index}")

        # Save token index
        token_index_path = "/home/ubuntu/lstm_text_generation/models/token_index.pkl"
        save_preprocessing_data(unique_token_index, token_index_path)
        loaded_token_index = load_preprocessing_data(token_index_path)
        # print(f"Loaded token index: {loaded_token_index}")

        print("\nCreating sequences...")
        N_WORDS = 5 # Example sequence length
        X_data, y_data = create_sequences(tokens, N_WORDS, unique_token_index)
        print(f"Shape of X: {X_data.shape}") # (num_sequences, n_words, vocab_size)
        print(f"Shape of y: {y_data.shape}") # (num_sequences, vocab_size)

        # Save other relevant info
        metadata = {
            "n_words": N_WORDS,
            "vocab_size": len(unique_tokens),
            "unique_tokens": unique_tokens
        }
        metadata_path = "/home/ubuntu/lstm_text_generation/models/metadata.pkl"
        save_preprocessing_data(metadata, metadata_path)
        loaded_metadata = load_preprocessing_data(metadata_path)
        # print(f"Loaded metadata: {loaded_metadata}")

    # Clean up dummy file
    # os.remove(dummy_data_path)
    print("\nPreprocessing example finished.")

