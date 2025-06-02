# LSTM Text Generation Project

This project implements an LSTM (Long Short-Term Memory) neural network for text generation. It learns patterns from a given text corpus and can generate new text sequences based on a starting seed.

## Overview

The core functionality involves:

1.  **Data Preprocessing**: Loading text data, tokenizing it into words, building a vocabulary, and creating fixed-length sequences of words as input features (X) and the next word as the target (y).
2.  **Model Training**: Building and training an LSTM model using TensorFlow/Keras on the prepared sequences. The trained model, along with necessary metadata (vocabulary, sequence length), is saved.
3.  **Text Generation**: Loading the trained model and metadata to generate new text based on a user-provided seed sequence.

## Project Structure

```
lstm_text_generation/
├── data/                 # Directory for input text data
│   └── (e.g., joined_text.txt) # Input corpus file (needs to be provided)
├── models/               # Directory for saved models and metadata
│   ├── lstm_text_gen_model.h5  # Saved trained model
│   ├── training_history.pkl    # Saved training history
│   ├── token_index.pkl         # Saved token-to-index mapping
│   └── metadata.pkl            # Saved metadata (vocab, sequence length)
├── scripts/              # Main executable scripts
│   ├── train.py          # Script to train the LSTM model
│   └── generate.py       # Script to generate text using a trained model
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py  # Data loading and preprocessing functions
│   ├── model.py          # LSTM model definition and training functions
│   └── generator.py      # Text generation functions
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── README.md             # This documentation file
└── requirements.txt      # Project dependencies
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd lstm_text_generation
    ```

2.  **Create a Python Environment (Recommended):**
    It's highly recommended to use a virtual environment (like `venv`) to manage dependencies.
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This includes TensorFlow, which might require specific system configurations (like CUDA for GPU support). Refer to the official TensorFlow installation guide if you encounter issues.*
    You will also need the NLTK data for tokenization:
    ```python
    import nltk
    nltk.download("punkt") # Required for tokenization
    ```
    Run this once in a Python interpreter after installing requirements.

4.  **Prepare Data:**
    *   This project requires a text corpus file for training.
    *   The reference notebook used `fake_or_real_news.csv`. You would typically process such a file to extract the text content into a single large text file.
    *   Place your training text file (e.g., `joined_text.txt`) inside the `data/` directory.
    *   **Important:** The `train.py` script expects the data file at `data/joined_text.txt` by default. You can change this using the `--data_path` argument.
    *   A sample command to potentially create `joined_text.txt` from the CSV (assuming the CSV is in the root and has a 'text' column):
        ```python
        # Example Python snippet (run separately or adapt)
        import pandas as pd
        df = pd.read_csv("fake_or_real_news.csv")
        text = " ".join(list(df.text.values))
        with open("data/joined_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        ```

## Usage

### 1. Training the Model

Run the `train.py` script to preprocess the data and train the LSTM model. The script will save the trained model (`.h5`), training history (`.pkl`), token index (`.pkl`), and metadata (`.pkl`) to the `models/` directory by default.

```bash
python scripts/train.py [OPTIONS]
```

**Key Options:**

*   `--data_path`: Path to your input text corpus (default: `data/joined_text.txt`).
*   `--text_limit`: Limit the number of characters to process for faster training (default: 1,000,000, use 0 for no limit).
*   `--epochs`: Number of training epochs (default: 10).
*   `--batch_size`: Training batch size (default: 128).
*   `--sequence_length`: Length of word sequences for model input (default: 10).
*   `--model_save_path`: Path to save the trained model (default: `models/lstm_text_gen_model.h5`).
*   *(See `scripts/train.py --help` for all options)*

**Example:**

```bash
# Train with default settings using data/joined_text.txt
python scripts/train.py

# Train for 20 epochs using a different data file
python scripts/train.py --data_path data/my_corpus.txt --epochs 20
```

### 2. Generating Text

Once the model is trained and saved, use the `generate.py` script to generate new word.

```bash
python scripts/generate.py --num_words=1 --seed_text="Your seed text goes here" [OPTIONS]
```
Or if to generate a creative text for Default length use
```bash
python scripts/generate.py "Your seed text goes here" [OPTIONS]
```

**Arguments:**

*   `seed_text` (Required): The initial sequence of words to start the generation. **Must contain at least `sequence_length` words** (default is 10). Enclose in quotes if it contains spaces.

**Key Options:**

*   `--model_path`: Path to the trained model file (default: `models/lstm_text_gen_model.h5`).
*   `--token_index_path`: Path to the token index file (default: `models/token_index.pkl`).
*   `--metadata_path`: Path to the metadata file (default: `models/metadata.pkl`).
*   `--num_words`: Number of words to generate after the seed text (default: 100).
*   `--creativity`: Controls randomness. Higher values consider more possible next words (default: 5).
*   *(See `scripts/generate.py --help` for all options)*

**Example:**

```bash
# Generate 50 words using the default model and a seed text
python scripts/generate.py "the president announced today that the new policy will focus on"

# Generate 200 words with higher creativity
python scripts/generate.py "it was a dark and stormy night the wind howled through the trees" --num_words 200 --creativity 10
```

## Dependencies

See `requirements.txt` for the list of Python packages required.

*   **TensorFlow**: For building and training the LSTM model.
*   **NLTK**: For text tokenization.
*   **NumPy**: For numerical operations.
*   **Pandas**: (Optional, useful for initial data loading/preparation if using CSV like the reference).

## Notes

*   Training LSTM models can be computationally intensive and time-consuming, especially on large datasets without GPU acceleration.
*   The quality of the generated text depends heavily on the size and quality of the training corpus, model architecture, and training duration.
*   The `text_limit` parameter in `train.py` is useful for quick testing on smaller subsets of your data.

