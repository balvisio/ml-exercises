import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        head_size_dim,
        dense_dim,
        num_heads,
        value_dim,
        key_value_seq_length=None,
        key_embedding_size=None,
        value_embedding_size=None,
        attn_output_shape=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.head_size_dim = head_size_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_value_seq_length = key_value_seq_length
        self.key_embedding_size = key_embedding_size
        self.value_embedding_size = value_embedding_size
        self.attn_output_shape = attn_output_shape
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size_dim,
            value_dim=value_dim,
            output_shape=attn_output_shape,
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(attn_output_shape),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        key = inputs
        value = inputs
        if self.key_value_seq_length:
            key = inputs[:, 0:self.key_value_seq_length, :]
            value = inputs[:, 0:self.key_value_seq_length, :]

        if self.key_embedding_size:
            key = key[:, :, 0:self.key_embedding_size]

        if self.value_embedding_size:
            value = value[:, :, 0:self.value_embedding_size]

        """
        The output of the "MultiHeadAttention" Layer or (equivalently) input to the “Dense” layer is
        a 3D matrix of dimensions (batch_size, query_sequence_length, word_embedding_size).
        There is a note in the Keras documentation for the Dense layer on how
        it handles tensors of rank greater than 2. (https://keras.io/api/layers/core_layers/dense/)
        Basically, it flattens the tensor to make it 2D, computes the matrix product with the kernel
        and then reshapes it back. In other words, it treats all dimensions except for the last one
        as if they were batches.
        """
        attention_output = self.attention(
            inputs, value, key=key, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_size_dim": self.head_size_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "value_dim": self.value_dim,
            "key_value_seq_length": self.key_value_seq_length,
            "key_embedding_size": self.key_embedding_size,
            "value_embedding_size": self.value_embedding_size,
            "attn_output_shape": self.attn_output_shape,
        })
        return config
