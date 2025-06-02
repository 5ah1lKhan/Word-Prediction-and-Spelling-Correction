```python
import random
import pickle
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
```


```python
text_df = pd.read_csv("fake_or_real_news.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)

with open("joined_text.txt", "w", encoding="utf-8") as f:
    f.write(joined_text)
```


```python
partial_text = joined_text[:1000000]
```


```python
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
```


```python
unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
```


```python
n_words = 10
input_words = []
next_word = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_word.append(tokens[i + n_words])
```


```python
X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)  # for each sample, n input words and then a boolean for each possible next word
y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)  # for each sample a boolean for each possible next word
```


```python
for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_word[i]]] = 1
```


```python
model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))
```


```python
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(X, y, batch_size=128, epochs=10, shuffle=True).history
```

    Epoch 1/10
    1326/1326 [==============================] - 347s 260ms/step - loss: 7.2461 - accuracy: 0.0604
    Epoch 2/10
    1326/1326 [==============================] - 335s 253ms/step - loss: 7.4470 - accuracy: 0.0883
    Epoch 3/10
    1326/1326 [==============================] - 349s 263ms/step - loss: 7.5690 - accuracy: 0.1006
    Epoch 4/10
    1326/1326 [==============================] - 336s 253ms/step - loss: 7.5587 - accuracy: 0.1083
    Epoch 5/10
    1326/1326 [==============================] - 343s 259ms/step - loss: 7.3973 - accuracy: 0.1187
    Epoch 6/10
    1326/1326 [==============================] - 339s 256ms/step - loss: 7.1945 - accuracy: 0.1297
    Epoch 7/10
    1326/1326 [==============================] - 339s 255ms/step - loss: 6.9307 - accuracy: 0.1443
    Epoch 8/10
    1326/1326 [==============================] - 341s 257ms/step - loss: 6.6039 - accuracy: 0.1638
    Epoch 9/10
    1326/1326 [==============================] - 342s 258ms/step - loss: 6.2665 - accuracy: 0.1870
    Epoch 10/10
    1326/1326 [==============================] - 339s 256ms/step - loss: 5.8945 - accuracy: 0.2125



```python
history = model.fit(X, y, batch_size=128, epochs=5, shuffle=True).history
```

    Epoch 1/5
    1326/1326 [==============================] - 421s 316ms/step - loss: 5.5365 - accuracy: 0.2450
    Epoch 2/5
    1326/1326 [==============================] - 428s 323ms/step - loss: 5.2118 - accuracy: 0.2751
    Epoch 3/5
    1326/1326 [==============================] - 426s 321ms/step - loss: 4.9456 - accuracy: 0.3037
    Epoch 4/5
    1326/1326 [==============================] - 427s 322ms/step - loss: 4.6771 - accuracy: 0.3346
    Epoch 5/5
    1326/1326 [==============================] - 426s 322ms/step - loss: 4.5058 - accuracy: 0.3530



```python
model.save("text_gen_model2.h5")
with open("history2.p", "wb") as f:
    pickle.dump(history, f)
```


```python
model = load_model("text_gen_model2.h5")
history = pickle.load(open("history2.p", "rb"))
```


```python
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
        
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]
```


```python
possible = predict_next_word("I will have to look into this thing because I", 5)
```


```python
for idx in possible:
    print(unique_tokens[idx])
```

    will
    can
    had
    don
    did



```python
def generate_text(input_text, n_words, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(n_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)
```


```python
generate_text("I will have to look into this thing because I", 100, 10)
```




    'I will have to look into this thing because I could see the we are president but could just more dubious from new deals a lower presidential bid the us president were just as william when she had his hands place in her state from national bureau it has been one with my party to will say it would could win all war us this and this was if you was sure the importance for a if that has made a lead against those people and what trump can t take their effect is that is that he is to be we will have president obama who s polling better'




```python
generate_text("The president of the United States announced yesterday that he", 100, 10)
```




    'The president of the United States announced yesterday that he won a president a president reached which cnn president a national stretch year implicit were also discuss before it one we would have a boost a man and raising the common part he would need for 50 and if i look a results but now this results who 50 about their ideas the go from systems if not that william when a us we can see all hollande to have served in american terrorist attacks the campaign was doing its foreign relations on their members are in a man is so under its you go a more control of clinton'




```python
for idx in predict_next_word("The president will most likely not be there to help", 5):
    print(unique_tokens[idx])
```

    american
    the
    our
    us
    president

