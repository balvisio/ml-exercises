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


"""
Output:

2022-07-11 22:39:48.439231: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the follow
ing CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-07-11 22:39:52.043586: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30466 MB memory:  -> device: 0, n
ame: Tesla V100S-PCIE-32GB, pci bus id: 0000:83:00.0, compute capability: 7.0
2022-07-11 22:39:52.046467: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 30404 MB memory:  -> device: 1, n
ame: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 english (InputLayer)           [(None, None)]       0           []

 spanish (InputLayer)           [(None, None)]       0           []

 positional_embedding (Position  (None, None, 456)   9479784     ['english[0][0]']
 alEmbedding)

 positional_embedding_1 (Positi  (None, None, 456)   9479784     ['spanish[0][0]']
 onalEmbedding)

 transformer_encoder (Transform  (None, None, 456)   575381      ['positional_embedding[0][0]']
 erEncoder)

 transformer_decoder (Transform  (None, None, 456)   765021      ['positional_embedding_1[0][0]',
 erDecoder)                                                       'transformer_encoder[0][0]']

 dropout (Dropout)              (None, None, 456)    0           ['transformer_decoder[0][0]']

 dense_4 (Dense)                (None, None, 20000)  9140000     ['dropout[0][0]']

==================================================================================================
Total params: 29,439,970
Trainable params: 29,439,970
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/30
2022-07-11 22:41:57.986115: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
1302/1302 [==============================] - 169s 123ms/step - loss: 1.5310 - accuracy: 0.4787 - val_loss: 1.2258 - val_accuracy: 0.5554
Epoch 2/30
1302/1302 [==============================] - 158s 121ms/step - loss: 1.2537 - accuracy: 0.5689 - val_loss: 1.1232 - val_accuracy: 0.5933
Epoch 3/30
1302/1302 [==============================] - 161s 124ms/step - loss: 1.1411 - accuracy: 0.6012 - val_loss: 1.0895 - val_accuracy: 0.6077
Epoch 4/30
1302/1302 [==============================] - 163s 125ms/step - loss: 1.0945 - accuracy: 0.6189 - val_loss: 1.0630 - val_accuracy: 0.6157
Epoch 5/30
1302/1302 [==============================] - 164s 126ms/step - loss: 1.0698 - accuracy: 0.6302 - val_loss: 1.0596 - val_accuracy: 0.6193
Epoch 6/30
1302/1302 [==============================] - 162s 125ms/step - loss: 1.0547 - accuracy: 0.6397 - val_loss: 1.0523 - val_accuracy: 0.6241
Epoch 7/30
1302/1302 [==============================] - 162s 124ms/step - loss: 1.0442 - accuracy: 0.6472 - val_loss: 1.0489 - val_accuracy: 0.6297
Epoch 8/30
1302/1302 [==============================] - 164s 126ms/step - loss: 1.0363 - accuracy: 0.6537 - val_loss: 1.0592 - val_accuracy: 0.6281
Epoch 9/30
1302/1302 [==============================] - 163s 125ms/step - loss: 1.0294 - accuracy: 0.6588 - val_loss: 1.0599 - val_accuracy: 0.6284
Epoch 10/30
1302/1302 [==============================] - 164s 126ms/step - loss: 1.0231 - accuracy: 0.6639 - val_loss: 1.0592 - val_accuracy: 0.6298
Epoch 11/30
1302/1302 [==============================] - 164s 126ms/step - loss: 1.0167 - accuracy: 0.6682 - val_loss: 1.0663 - val_accuracy: 0.6311
Epoch 12/30
1302/1302 [==============================] - 160s 123ms/step - loss: 1.0106 - accuracy: 0.6719 - val_loss: 1.0656 - val_accuracy: 0.6337
Epoch 13/30
1302/1302 [==============================] - 164s 126ms/step - loss: 1.0048 - accuracy: 0.6751 - val_loss: 1.0794 - val_accuracy: 0.6300
Epoch 14/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9988 - accuracy: 0.6783 - val_loss: 1.0754 - val_accuracy: 0.6292
Epoch 15/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9922 - accuracy: 0.6815 - val_loss: 1.0741 - val_accuracy: 0.6295
Epoch 16/30
1302/1302 [==============================] - 164s 126ms/step - loss: 0.9860 - accuracy: 0.6843 - val_loss: 1.0795 - val_accuracy: 0.6299
Epoch 17/30
1302/1302 [==============================] - 162s 125ms/step - loss: 0.9800 - accuracy: 0.6872 - val_loss: 1.0789 - val_accuracy: 0.6325
Epoch 18/30
1302/1302 [==============================] - 164s 126ms/step - loss: 0.9727 - accuracy: 0.6901 - val_loss: 1.0809 - val_accuracy: 0.6320
Epoch 19/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9671 - accuracy: 0.6926 - val_loss: 1.0853 - val_accuracy: 0.6321
Epoch 20/30
1302/1302 [==============================] - 164s 126ms/step - loss: 0.9614 - accuracy: 0.6950 - val_loss: 1.0865 - val_accuracy: 0.6326
Epoch 21/30
1302/1302 [==============================] - 164s 126ms/step - loss: 0.9543 - accuracy: 0.6977 - val_loss: 1.0888 - val_accuracy: 0.6312
Epoch 22/30
1302/1302 [==============================] - 162s 125ms/step - loss: 0.9495 - accuracy: 0.6996 - val_loss: 1.0934 - val_accuracy: 0.6313
Epoch 23/30
1302/1302 [==============================] - 162s 124ms/step - loss: 0.9441 - accuracy: 0.7019 - val_loss: 1.0961 - val_accuracy: 0.6310
Epoch 24/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9397 - accuracy: 0.7035 - val_loss: 1.1002 - val_accuracy: 0.6300
Epoch 25/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9347 - accuracy: 0.7053 - val_loss: 1.1023 - val_accuracy: 0.6289
Epoch 26/30
1302/1302 [==============================] - 162s 125ms/step - loss: 0.9303 - accuracy: 0.7071 - val_loss: 1.1002 - val_accuracy: 0.6326
Epoch 27/30
1302/1302 [==============================] - 162s 124ms/step - loss: 0.9248 - accuracy: 0.7087 - val_loss: 1.1044 - val_accuracy: 0.6313
Epoch 28/30
1302/1302 [==============================] - 163s 125ms/step - loss: 0.9197 - accuracy: 0.7106 - val_loss: 1.1053 - val_accuracy: 0.6310
Epoch 29/30
1302/1302 [==============================] - 162s 124ms/step - loss: 0.9151 - accuracy: 0.7119 - val_loss: 1.1065 - val_accuracy: 0.6307
Epoch 30/30
1302/1302 [==============================] - 160s 123ms/step - loss: 0.9105 - accuracy: 0.7140 - val_loss: 1.1166 - val_accuracy: 0.6290

"""