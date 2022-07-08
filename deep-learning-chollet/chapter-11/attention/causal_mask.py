import tensorflow as tf

def get_causal_attention_mask(inputs):
    input_shape = tf.shape(inputs)
    print(f"Input shape: {input_shape}")
    batch_size, sequence_length = input_shape[0], input_shape[1]
    print(f"Batch Size: {batch_size}")
    i = tf.range(sequence_length)[:, tf.newaxis]
    print(f"i shape: {i.shape}")
    print(i)
    j = tf.range(sequence_length)
    print(f"j shape: {j.shape}")
    print(j)
    """
    tf.cast casts a tensor to a new type.
    """
    mask = tf.cast(i >= j, dtype = "int32")
    print(f"Mask shape: {mask.shape}")
    print(mask)
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    print(f"Mask reshaped: {mask.shape}")
    print(mask)
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
        tf.constant([1, 1], dtype=tf.int32)], axis=0)
    print(tf.expand_dims(batch_size, -1))
    print(f"Mult: {mult}")
    """
    tf.tile creates a new tensor by replicating input multiples times.
    tf.tile(
        input, multiples, name=None
    )
    """
    res = tf.tile(mask, mult)
    print(res)
    
input = [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]]

get_causal_attention_mask(input)
