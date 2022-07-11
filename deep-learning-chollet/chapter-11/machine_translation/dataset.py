"""
This code is to prepare a Dataset for an English-to-Spanish translator model.

The dataset was downloaded from http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
"""
import pathlib
import random
import tensorflow as tf
from tensorflow.keras import layers
import string
import re

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


def create_text_pairs():
    text_file = pathlib.Path.home() / "spa-eng/spa.txt"

    with open(text_file) as f:
        lines = f.read().split("\n")[:-1]

    text_pairs = []

    for line in lines:
        english, spanish = line.split("\t")
        spanish = "[start] " + spanish + " [end]"
        text_pairs.append((english, spanish))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]

    return train_pairs, val_pairs, test_pairs


def custom_standarization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


def create_text_vectorization(pairs, vocab_size):
    sequence_length = 20

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standarization,
    )

    train_english_texts = [pair[0] for pair in pairs]
    train_spanish_texts = [pair[1] for pair in pairs]

    source_vectorization.adapt(train_english_texts)
    target_vectorization.adapt(train_spanish_texts)

    return source_vectorization, target_vectorization


def format_dataset(eng, spa, source_vectorization, target_vectorization):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)

    # Returns a the tuple (input, target) where inputs is a dict with two keys:
    # - "encoder_inputs": english sentence
    # - "decoder_inputs": spanish sentenct
    #
    # and the target is the spanish sentence offset by one step ahead.
    return ({
        "english": eng,
        "spanish": spa[:, :-1]
    }, spa[:, 1:])


def make_dataset(pairs, source_vectorization, target_vectorization):
    batch_size = 64

    """
    Difference between zip(list) and zip(*list):

    https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist

    type(eng_texts) == <class 'tuple'>
    """
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: format_dataset(x, y, source_vectorization, target_vectorization), num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()


if __name__ == "__main__":
    vocab_size = 15000
    train_pairs, val_pairs, test_pairs = create_text_pairs()
    source_vectorization, target_vectorization = create_text_vectorization(train_pairs, vocab_size)

    train_ds = make_dataset(train_pairs, source_vectorization, target_vectorization)
    val_ds = make_dataset(val_pairs, source_vectorization, target_vectorization)

    for inputs, targets in train_ds.take(1):
        print(f"inputs['english'].shape: {inputs['english'].shape}")
        print(f"Sample: {inputs['english'][0]}")
        print(f"inputs['spanish'].shape: {inputs['english'].shape}")
        print(f"targets.shape: {targets.shape}")

        """
        inputs['english'].shape: (64, 20)
        Sample: [   6 1624  259   12    9  416   30  221   12    0    0    0    0    0
            0    0    0    0    0    0]
        inputs['spanish'].shape: (64, 20)
        targets.shape: (64, 20)
        """