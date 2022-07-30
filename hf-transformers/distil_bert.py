import tensorflow as tf
import transformers
from transformers import TFDistilBertModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = TFDistilBertModel.from_pretrained(
    "distilbert-base-cased",
    output_hidden_states=True,
    output_attentions=True,
)
sequence = (
    "Distilled models are smaller than the models they mimic. Using them instead of the large "
    f"versions would help {tokenizer.mask_token} our carbon footprint"
)

inputs = tokenizer(sequence, return_tensors="tf")

output = model(**inputs)

model.summary()

print(type(output)) # <class 'transformers.modeling_tf_outputs.TFBaseModelOutput'>
"""
From https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput:
the output is a tuple with elements:

- last_hidden_state: (batch_size, sequence_length, hidden_size)

- hidden_states: Hidden-states of the model at the output of each layer plus the initial embedding 
outputs. It is a tuple of length: 1 (embeddings) + # of hidden layers.
The shape of each element is (batch_size, sequence_length, hidden_size).
The default number of layers of DistilBert is 6. See https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/distilbert#transformers.DistilBertConfig

- attentions: Attentions weights after the attention softmax, used to compute the weighted 
average in the self-attention heads.
It is a tuple of length: # of hidden layers.
The shape of each element is (batch_size, num_heads, sequence_length, sequence_length).

NOTE: The output of the of the DistilBertModel is different from the BertModel since the latter one
outputs "TFBaseModelOutputWithPoolingAndCrossAttentions" which contains the: "last_hidden_layers"
and the "pooler_outputs". For more info see: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions
"""

print(len(output))
"""
When output_hidden_states=False => len(output) == 1
When output_hidden_states=True => len(output) == 2
When output_hidden_states=True & output_attentions=True => len(output) == 3

"""

print(output[0].shape)
"""
Shape of last_hidden_state is (batch_size, sequence_length, hidden_size).
In our case: (1, 29, 768)
"""

# Hidden States
print(len(output[1])) # 1 (embeddings) + # of hidden layers = 7

print(output[1][0].shape)
"""
Shape of hidden_states is (batch_size, sequence_length, hidden_size).
In our case: (1, 29, 768)
"""

# Check that the last_hidden_state is identical that taking the last element from hidden_states
assert(tf.math.reduce_all(tf.equal(output[0],output[1][-1])) == True)

# Attentions
print(len(output[2])) # # of hidden layers = 6

print(output[2][0].shape)
"""
Shape of attentions is (batch_size, num_heads, sequence_length, sequence_length).
In our case: (1, 12, 29, 29)
"""

"""
The following code is just to showcase how the model fails if we pass a sequence long than 512.

This is because distil-bert was built with a positional embedding layer of maximum size 512.
This can be seen in the param "max_position_embeddings" of its config:
https://huggingface.co/distilbert-base-cased/blob/main/config.json

If we pass a sequence longer than 512 the embedding layer cannot handle anything equal or greater
than 512. Error:

indices[0,512] = 512 is not in [0, 512) [Op:ResourceGather]

Call arguments received by layer "embeddings" (type TFEmbeddings):
  • input_ids=tf.Tensor(shape=(1, 513), dtype=int32)
"""
inputs = tokenizer(sequence, return_tensors="tf", padding=True)

my_dict = {
    "input_ids": tf.constant(1, shape=(1,513), name='Const'),
    "attention_mask": tf.constant(1, shape=(1,513), name='Const'),
}

inputs = transformers.tokenization_utils_base.BatchEncoding(my_dict)
output = model(**inputs)
