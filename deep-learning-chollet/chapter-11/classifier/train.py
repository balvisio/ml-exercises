"""
The following code is based on the the book "Deep Learning with Python" by Francois Chollet (2nd
edition), Chapter 11, Section 4.

The code has been annotatated and modified to be more clear and more flexible.
"""

import os, pathlib, shutil, random, sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from transformer.transformer_encoder import TransformerEncoder


def create_ds():
    base_dir = pathlib.Path.home() / pathlib.Path("aclImdb")
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"

    if not os.path.isdir(val_dir):
        print("Building validation directory")
        for category in ("neg", "pos"):
            os.makedirs(val_dir / category)
            files = os.listdir(train_dir / category)
            random.Random(1337).shuffle(files)
            num_val_samples = int(0.2 * len(files))
            val_files = files[-num_val_samples:]
            for fname in val_files:
                shutil.move(train_dir / category / fname,
                            val_dir / category / fname)
    else:
        print("Skipping building validation dir")

    batch_size = 32
    
    train_ds = keras.utils.text_dataset_from_directory(
        base_dir / "train", batch_size=batch_size
    )
    
    val_ds = keras.utils.text_dataset_from_directory(
        base_dir / "val", batch_size=batch_size
    )

    test_ds = keras.utils.text_dataset_from_directory(
        base_dir / "test", batch_size=batch_size
    )

    return train_ds, val_ds, test_ds


def create_int_ds(train_ds, val_ds, test_ds):
    max_sequence_length = 789
    max_tokens = 20000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_sequence_length,
    )
    
    text_only_train_ds = train_ds.map(lambda x, y: x)
    """
    During adapt(), the layer will build a vocabulary of all string tokens seen in the dataset,
    sorted by occurance count, with ties broken by sort order of the tokens (high to low).
    At the end of adapt(), if max_tokens is set, the vocabulary wil be truncated to max_tokens size.
    """
    text_vectorization.adapt(text_only_train_ds)
    int_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    """
    int_train_ds: It is a tuple of shape:
    (<tf.Tensor: shape=(batch_size, max_sequence_length), dtype=int64>, <<tf.Tensor: shape=(batch_size,), dtype=int32>)
    """
    int_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    int_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

    return int_train_ds, int_val_ds, int_test_ds

train_ds, val_ds, test_ds = create_ds()
int_train_ds, int_val_ds, int_test_ds = create_int_ds(train_ds, val_ds, test_ds)

tf.data.experimental.save(int_test_ds, "int_test_ds")
    
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
use_mask = False

"""
This optional parameter to the encoder is to showcase that the sequence length of the query
doesn't necessarily need to be equal to the query sequence length.
"""
key_value_seq_length = 342

"""
These optional parameters to the encoder are to showcase that technically the embedding dimensions
of the K, Q, V can be all different. The matrices multiplications work because remember that the
K, Q, V input matrices go through the inner dense blocks before the attention is computed.
"""
key_embedding_size = 131
value_embedding_size = 132

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, word_embed_dim, mask_zero=use_mask)(inputs)
x = TransformerEncoder(
    head_size_dim,
    dense_dim,
    num_heads,
    value_dim,
    key_value_seq_length,
    key_embedding_size,
    value_embedding_size,
    output_shape,
)(x)
"""
The output of the "TransformerEncoder" is a 3D matrix of dimensions: (batch_size,
query_sequence_length, word_embedding_size).
Since TransformerEncoder returns full sequences, we need to reduce each sequence to a single vector
for classification via a global pooling layer. From each 'word' in the sequence we take the maximum
value along the embedding. The resulting matrix is of size: (batch_size, word_embedding).
"""
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True,
    )
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "transformer_encoder.keras",
        save_best_only=True,
    )
]

model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=20,
    callbacks=callbacks
)