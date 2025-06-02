# -*- coding: utf-8 -*-
"""Text Generation Logic using a trained LSTM model."""

import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf

# Ensure TF doesn't allocate all GPU memory if a GPU is present
try:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"Could not configure GPU memory growth: {e}")

def predict_next_token_indices(model, input_sequence_vector, n_best=1):
    """Predicts the indices of the most likely next tokens.

    Args:
        model (tf.keras.Model): The trained Keras LSTM model.
        input_sequence_vector (np.ndarray): The vectorized input sequence 
                                            (shape: 1, sequence_length, vocab_size).
        n_best (int): The number of top predictions to return.

    Returns:
        np.ndarray: An array containing the indices of the n_best predicted tokens.
    """
    predictions = model.predict(input_sequence_vector, verbose=0)[0] # Get predictions for the single sequence
    # Use argpartition for efficiency if n_best is small compared to vocab_size
    # It finds the k smallest elements; use negative predictions to find k largest.
    # Ensure n_best does not exceed vocab size
    n_best = min(n_best, len(predictions))
    # Get indices of the top n_best predictions
    best_indices = np.argpartition(predictions, -n_best)[-n_best:]
    # Sort these indices by probability (descending)
    best_indices_sorted = best_indices[np.argsort(predictions[best_indices])[::-1]]
    return best_indices_sorted

def generate_text(model, token_index, unique_tokens, seed_text, sequence_length, num_words_to_generate, creativity=3):
    """Generates text by iteratively predicting the next word.

    Args:
        model (tf.keras.Model): The trained Keras LSTM model.
        token_index (dict): Dictionary mapping tokens to their indices.
        unique_tokens (list): List of unique tokens (vocabulary).
        seed_text (str): The initial text sequence to start generation.
        sequence_length (int): The length of sequences the model expects (n_words).
        num_words_to_generate (int): The number of words to generate after the seed text.
        creativity (int): How many top predictions to consider for random choice (higher means more randomness).

    Returns:
        str: The generated text including the seed text.
    """
    vocab_size = len(unique_tokens)
    tokenizer = RegexpTokenizer(r"\w+")
    word_sequence = tokenizer.tokenize(seed_text.lower())

    if len(word_sequence) < sequence_length:
        print(f"Warning: Seed text has {len(word_sequence)} words, but model expects sequence length {sequence_length}. Padding or adjusting might be needed, or provide longer seed.")
        # Basic handling: return seed if too short for now
        # A more robust solution might involve padding or erroring
        # For now, we'll try to proceed but it might fail if the model requires exact length
        # Let's try padding with a known token if possible, or just truncate if too long
        # Simplest: require seed >= sequence_length
        if len(word_sequence) == 0:
             print("Error: Seed text is empty.")
             return seed_text # Or raise error
        # Pad with the first word if needed (very basic strategy)
        # while len(word_sequence) < sequence_length:
        #     word_sequence.insert(0, word_sequence[0]) 
        print(f"Error: Seed text length ({len(word_sequence)}) must be at least sequence length ({sequence_length}).")
        return seed_text # Return original seed text if too short

    current_sequence_tokens = word_sequence[-sequence_length:]

    generated_text = list(word_sequence) # Start with the seed sequence

    for _ in range(num_words_to_generate):
        # Vectorize the current sequence
        X_pred = np.zeros((1, sequence_length, vocab_size), dtype=bool)
        for i, word in enumerate(current_sequence_tokens):
            if word in token_index:
                X_pred[0, i, token_index[word]] = 1
            # else: word not in vocab, vector remains zero for this position

        # Predict the next token indices
        predicted_indices = predict_next_token_indices(model, X_pred, n_best=creativity)

        if len(predicted_indices) == 0:
            # Handle case where prediction fails or returns nothing (shouldn't happen with softmax)
            print("Warning: Prediction returned no indices. Choosing random token.")
            next_token = random.choice(unique_tokens)
        else:
            # Choose one index randomly from the top 'creativity' predictions
            chosen_index = random.choice(predicted_indices)
            next_token = unique_tokens[chosen_index]

        # Append the chosen token and update the sequence
        generated_text.append(next_token)
        current_sequence_tokens = generated_text[-sequence_length:]

    return " ".join(generated_text)

# Example usage (for demonstration)
if __name__ == "__main__":
    print("Generator module example (requires a trained model and metadata)")
    # This part would normally load a model and metadata
    # For demonstration, we'll assume dummy values

    # Dummy model (replace with actual loaded model)
    class DummyModel:
        def predict(self, x, verbose=0):
            # Simulate prediction output shape (batch_size, vocab_size)
            # Return random probabilities
            return np.random.rand(1, 50) # Assume vocab_size=50

    dummy_model = DummyModel()
    dummy_token_index = {f"word{i}": i for i in range(50)}
    dummy_unique_tokens = [f"word{i}" for i in range(50)]
    dummy_seq_len = 5

    seed = "word0 word1 word2 word3 word4"
    print(f"Seed text: {seed}")

    # Test prediction function
    X_dummy = np.zeros((1, dummy_seq_len, 50), dtype=bool)
    for i, word in enumerate(seed.split()):
        if word in dummy_token_index:
            X_dummy[0, i, dummy_token_index[word]] = 1
    
    pred_indices = predict_next_token_indices(dummy_model, X_dummy, n_best=5)
    print(f"Predicted next token indices (dummy): {pred_indices}")
    if len(pred_indices) > 0:
      print(f"Predicted next tokens (dummy): {[dummy_unique_tokens[i] for i in pred_indices]}")

    # Test generation function
    generated = generate_text(dummy_model, dummy_token_index, dummy_unique_tokens, seed, dummy_seq_len, 20, creativity=5)
    print(f"\nGenerated text (dummy):\n{generated}")

    print("\nGenerator module example finished.")

