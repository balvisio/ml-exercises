import sys

import json
import numpy as np
import pickle
import random

from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[2]
sys.path.append(str(package_root_directory))

from transformer.positional_embedding import PositionalEmbedding
from transformer.transformer_decoder import TransformerDecoder
from transformer.transformer_encoder import TransformerEncoder


def decode_sequence(
    input_sentence,
    source_vectorization,
    target_vectorization,
    spa_index_lookup,
    transformer,
):
    max_decoded_sequence_length = 20
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"

    for i in range(max_decoded_sequence_length):
        """
        Differing from the RNN case, in the Transformer the "tokenized_input_sentence" and the
        "tokenized_target_sentence" need to be of the same length. Since the decoded_sentence has
        shape (1, 21) the last element is removed to make it (1, 20)
        """
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence]
        )
        sample_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sample_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


def main():
    transformer = keras.models.load_model(
        "transformer.keras",
        custom_objects={
            "TransformerEncoder": TransformerEncoder,
            "TransformerDecoder": TransformerDecoder,
            "PositionalEmbedding": PositionalEmbedding,
        }
    )

    source_vectorization_ser = pickle.load(open("source_vectorization.pkl", "rb"))
    source_vectorization = layers.TextVectorization.from_config(source_vectorization_ser["config"])
    source_vectorization.set_weights(source_vectorization_ser["weights"])

    target_vectorization_ser = pickle.load(open("target_vectorization.pkl", "rb"))
    target_vectorization = layers.TextVectorization.from_config(target_vectorization_ser["config"])
    target_vectorization.set_weights(target_vectorization_ser["weights"])

    with open("test_pairs.json", "r") as f:
        test_pairs = json.load(f)

    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    test_eng_texts = [pair[0] for pair in test_pairs]
    for _ in range(20):
        input_sentence = random.choice(test_eng_texts)
        print("-")
        print(input_sentence)
        print(decode_sequence(
            input_sentence,
            source_vectorization,
            target_vectorization,
            spa_index_lookup,
            transformer,
        ))


if __name__ == "__main__":
    main()