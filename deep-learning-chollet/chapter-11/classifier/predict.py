"""
The following code is based on the the book "Deep Learning with Python" by Francois Chollet (2nd
edition), Chapter 11, Section 4.

The code has been annotatated and modified to be more clear and more flexible.
"""
import sys

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from transformer.positional_embedding import PositionalEmbedding
from transformer.transformer_encoder import TransformerEncoder

text_vectorization_ser = pickle.load(open("text_vectorization.pkl", "rb"))
text_vectorization = layers.TextVectorization.from_config(text_vectorization_ser["config"])
text_vectorization.set_weights(text_vectorization_ser["weights"])

vocab = text_vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
word_lookup = dict((v, k) for k, v in index_lookup.items())

model = keras.models.load_model(
    "transformer_encoder.keras",
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding,
    },
)

# Predict sample from test dataset
int_test_ds = tf.data.experimental.load("int_test_ds")

batch_tokens, labels = next(iter(int_test_ds))
element = batch_tokens[0]
new_batch = tf.expand_dims(element, 0)

decoded_sentence = ""
for index in element.numpy():
    if index == 0:
        break
    decoded_sentence += " " + index_lookup[index]

print(f"Input sentence: {decoded_sentence}")
print(f"Prediction score: {model.predict(new_batch)[0][0]:.3f}")

# Predict from sentences. We do manual tokenzation to show that transformer can handle
# sequences of different lengths.
sentence = "the movie was awesome"

tokenized_sentence = []
for word in sentence.split(" "):
    tokenized_sentence.append(word_lookup[word])

tokenized_sentence = tf.convert_to_tensor([tokenized_sentence])
print(f"Input sentence: {sentence}. Tokenized_sentence: {tokenized_sentence}")
print(f"Prediction score: {model.predict(tokenized_sentence)[0][0]:.3f}")
