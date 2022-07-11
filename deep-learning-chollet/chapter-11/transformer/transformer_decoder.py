import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerDecoder(layers.Layer):
    def __init__(
        self,
        query_and_key_head_size_dim,
        dense_dim,
        num_heads,
        attn_output_shape,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.query_and_key_head_size_dim = query_and_key_head_size_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attn_output_shape = attn_output_shape
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=query_and_key_head_size_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=query_and_key_head_size_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(attn_output_shape),
            ]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "query_and_key_head_size_dim": self.query_and_key_head_size_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            "attn_output_shape": self.attn_output_shape
        })

        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype = "int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32)], axis=0)
        """
        tf.tile creates a new tensor by replicating input multiples times. 
        tf.tile(
            input, multiples, name=None
        )
        """
        return tf.tile(mask, mult)
    
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32"
            )
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
        )
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2
        )
        proj_output = self.dense_proj(attention_output_2)  
        return self.layernorm_3(attention_output_2 + proj_output)
