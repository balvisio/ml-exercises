import sys

import json
import pickle
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[2]
sys.path.append(str(package_root_directory))

from machine_translation import dataset

from transformer.positional_embedding import PositionalEmbedding
from transformer.transformer_decoder import TransformerDecoder
from transformer.transformer_encoder import TransformerEncoder


def get_model(
    max_sequence_length,
    vocab_size,
    word_embed_dim,
    head_size_dim,
    dense_dim,
    num_heads,
    value_dim,
    output_shape,
):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
    x = PositionalEmbedding(max_sequence_length, vocab_size, word_embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(
        head_size_dim=head_size_dim,
        dense_dim=dense_dim,
        num_heads=num_heads,
        value_dim=value_dim,
        attn_output_shape=output_shape,
    )(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
    x = PositionalEmbedding(max_sequence_length, vocab_size, word_embed_dim)(decoder_inputs)
    x = TransformerDecoder(
        query_and_key_head_size_dim=head_size_dim,
        dense_dim=dense_dim,
        num_heads=num_heads,
        attn_output_shape=output_shape,
    )(x, encoder_outputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return transformer


def main():
    """
    "max_sequence_length" is equivalent to the query sequence length
    """
    max_sequence_length = 789

    """
    "key_value_seq_length" is optional parameter to the encoder is to showcase that the sequence length
    of the query doesn't necessarily need to be equal to the query sequence length.

    HOWEVER, when doing self-attention and passing a mask, the mask should be of size
    (batch_size, query_seq_length, key_seq_length). The "Embeding" and "PositionalEmbedding" layers
    generate a mask of size (batch_size, query_seq_length) and inside the "TransformerEncoder" layer
    the mask is transformed by adding an extra axis to (batch_size, 1, query_seq_length). The implicit
    assumption is that the key and query sequence length need to be the same. This is a reasonable
    assumption when doing self-attention since each element of the sequence is a word and the objective
    is to compute how much "attention" a word should pay to each of the other words.

    This mask is then broadcasted to be a tensor of shape (batch_size, query_seq_length,
    key_seq_length). Thus, for masking to work "max_sequence_length (query_seq_length)" has to be equal
    to "key_value_seq_length (query_seq_length)"

    Another option would be to do more manipulation of the mask tensor inside the TransformerEncoder
    layer.

    For more info see OneNote:NLP:Attention
    """
    key_value_seq_length = 789

    vocab_size = 20000

    """
    Originally called "embed_dim", the "head_size_dim" translates to the "key_dim" of the
    "MultiHeadAttention" and it represents the size of each attention head for query and key
    """
    head_size_dim = 97
    num_heads = 2

    """
    "word_embed_dim" is the size of the embedding for the Embedding layer.
    "output_shape" defines the size the output dimension of the "MultiHeadAttention" layer. Technically,
    this parameter doesn't need to be equal to "word_embed_dim" but if not, the "TransformerEncoder"
    will give an error when it tries to do the layer 1 normalization
    """
    word_embed_dim = 456
    output_shape = 456

    """
    This parameter is for the dimension of the Dense layers that come "after" and outside
    the "MultiHeadAttention" layer
    """
    dense_dim = 57
    value_dim = 188 # This is the size of the head of Value (V). Defaults to "key_dim/embed_dim"

    train_pairs, val_pairs, test_pairs = dataset.create_text_pairs()
    source_vectorization, target_vectorization = dataset.create_text_vectorization(
        train_pairs,
        vocab_size
    )

    # Save source and target vectorization layers so that they can be used during inference
    pickle.dump(
        {
            "config": source_vectorization.get_config(),
            "weights": source_vectorization.get_weights(),
        },
        open("source_vectorization.pkl", "wb"),
    )

    pickle.dump(
        {
            "config": target_vectorization.get_config(),
            "weights": target_vectorization.get_weights(),
        },
        open("target_vectorization.pkl", "wb"),
    )

    # Save test pairs for prediction
    with open("test_pairs.json", "w") as f:
        json.dump(test_pairs, f)

    train_ds = dataset.make_dataset(train_pairs, source_vectorization, target_vectorization)
    val_ds = dataset.make_dataset(val_pairs, source_vectorization, target_vectorization)

    transformer = get_model(
        max_sequence_length,
        vocab_size,
        word_embed_dim,
        head_size_dim,
        dense_dim,
        num_heads,
        value_dim,
        output_shape,
    )

    transformer.summary()

    transformer.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "transformer.keras",
            save_best_only=True,
        )
    ]

    transformer.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=callbacks)


if __name__ == "__main__":
    main()