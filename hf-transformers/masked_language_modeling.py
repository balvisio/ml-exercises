from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

sequence = (
    "Distilled models are smaller than the models they mimic. Using them instead of the large "
    f"versions would help {tokenizer.mask_token} our carbon footprint"
)

inputs = tokenizer(sequence, return_tensors="tf")

print(type(inputs)) # <class 'transformers.tokenization_utils_base.BatchEncoding'>

"""
BatchEncoding __dict__

inputs = {
    "data": {
        "input_ids": [ ... ],
        "attention_mask": [ ... ]
    },
    "_encodings": [ ... ],
    "_n_sequences": int
}
"""

print(tokenizer.mask_token_id) # 103

"""
in "tf.where" When only condition is provided the result is an int64 tensor where each row is 
the index of a non-zero element of condition. The result's shape is:

[tf.math.count_nonzero(condition), tf.rank(condition)].

In our case the result is: 

tf.Tensor([[ 0 23]], shape=(1, 2), dtype=int64)

because there is 1 element that is equal to the mask_token_id and in it is in a rank-2 tensor
(batch_size, seq_length). By selecting [0, 1] we are picking the first element found (0) and we are
picking the second dimension of the tensor (seq_length)
"""
mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1] # 23

token_logits = model(**inputs).logits
"""
token_logits.shape = (batch_size, seq_length, vocab_size) == (1, 29, 28996)
"""
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))