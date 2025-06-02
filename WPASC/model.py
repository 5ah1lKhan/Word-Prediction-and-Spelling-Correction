# -*- coding: utf-8 -*-
"""LSTM Model Definition, Training, and Saving/Loading Utilities"""

import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

def build_lstm_model(sequence_length, vocab_size):
    """Builds the LSTM model architecture based on the reference notebook.

    Args:
        sequence_length (int): The length of the input sequences (n_words).
        vocab_size (int): The total number of unique tokens in the vocabulary.

    Returns:
        tensorflow.keras.models.Sequential: The compiled Keras model.
    """
    model = Sequential()
    # Using 128 units as in the reference notebook
    model.add(LSTM(128, input_shape=(sequence_length, vocab_size), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))

    # Using RMSprop optimizer with learning rate 0.01 as in the reference
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("Model Summary:")
    model.summary()
    return model

def train_model(model, X, y, batch_size=128, epochs=10, shuffle=True):
    """Trains the LSTM model.

    Args:
        model (tensorflow.keras.models.Sequential): The compiled Keras model.
        X (np.ndarray): Input sequence data.
        y (np.ndarray): Target word data.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        shuffle (bool): Whether to shuffle the data before each epoch.

    Returns:
        tensorflow.keras.callbacks.History: The training history object.
    """
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    print("Training finished.")
    return history.history # Return the history dictionary

def save_model_and_history(model, history, model_path, history_path):
    """Saves the trained model and its training history.

    Args:
        model (tensorflow.keras.models.Sequential): The trained Keras model.
        history (dict): The training history dictionary.
        model_path (str): Path to save the model (.h5 file).
        history_path (str): Path to save the history (pickle file).
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        model.save(model_path)
        print(f"Model saved to {model_path}")

        with open(history_path, "wb") as f:
            pickle.dump(history, f)
        print(f"History saved to {history_path}")
    except Exception as e:
        print(f"Error saving model or history: {e}")

def load_trained_model(model_path):
    """Loads a pre-trained Keras model.

    Args:
        model_path (str): Path to the saved model (.h5 file).

    Returns:
        tensorflow.keras.models.Sequential: The loaded Keras model, or None if error.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# Example usage (for demonstration)
if __name__ == "__main__":
    # Dummy data parameters (replace with actual data loading in train script)
    SEQ_LENGTH = 10
    VOCAB_SIZE = 5000
    NUM_SAMPLES = 10000

    print("Building model...")
    model = build_lstm_model(SEQ_LENGTH, VOCAB_SIZE)

    # Create dummy training data
    print("\nGenerating dummy data for demonstration...")
    dummy_X = np.random.rand(NUM_SAMPLES, SEQ_LENGTH, VOCAB_SIZE) > 0.9
    dummy_y = np.random.rand(NUM_SAMPLES, VOCAB_SIZE) > 0.99
    # Ensure at least one true value per row in y for categorical crossentropy
    for i in range(NUM_SAMPLES):
        if not np.any(dummy_y[i]):
            dummy_y[i, np.random.randint(0, VOCAB_SIZE)] = True

    print("Training model on dummy data (1 epoch)...")
    # Train for only 1 epoch for quick demonstration
    history_data = train_model(model, dummy_X, dummy_y, epochs=1)
    print(f"Dummy training history: {history_data}")

    print("\nSaving model and history...")
    model_save_path = "/home/ubuntu/lstm_text_generation/models/dummy_model.h5"
    history_save_path = "/home/ubuntu/lstm_text_generation/models/dummy_history.pkl"
    save_model_and_history(model, history_data, model_save_path, history_save_path)

    print("\nLoading model...")
    loaded_model = load_trained_model(model_save_path)

    if loaded_model:
        print("Model loaded successfully.")
        loaded_model.summary()

    print("\nModel module example finished.")

