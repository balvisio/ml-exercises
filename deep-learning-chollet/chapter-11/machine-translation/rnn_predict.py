import json
import numpy as np
import pickle
import random
from tensorflow import keras
from tensorflow.keras import layers


def decode_sequence(
    model,
    source_vectorization,
    target_vectorization,
    spa_index_lookup,
    input_sentence
):
    max_decoded_sentence_length = 20
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = model.predict(
            [tokenized_input_sentence, tokenized_target_sentence]
        )
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


if __name__ == "__main__":
    seq2seq_rnn = keras.models.load_model(
        "seq2seq_rnn.keras",
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
            seq2seq_rnn,
            source_vectorization,
            target_vectorization,
            spa_index_lookup,
            input_sentence,
        ))
