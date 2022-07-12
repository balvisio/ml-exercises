import sys

import json
import numpy as np
import pickle
import random
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[2]
sys.path.append(str(package_root_directory))


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
        """
        The "tokenized_target_sentence.shape" is (1, 21) == (1, sequence_length + 1)
        Note that decoded_sentence is enclosed by [] to put it in "batch" format
        In the first iteration of the loop, tokenized_target_sentence == [[2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
        where 2 == [start]

        The length of tokenized_target_sentence is determined by the parameters used when building
        the "TextVectorization" layer. In "translation.py", the "target_vectorization" layer is
        built with "output_sequence_length=sequence_length + 1" (== 20 + 1).

        The length of "tokenized_target_sentence" are the number of "timesteps" that the RNN will
        be executed; in this case, 21. Thus, given that the decoder RNN was built with
        "return_sequences=True": the shape of 'next_token_predictions' is:
        (batch_size, sequence_length + 1, vocab_size)
        In out code example: (1, 21, 15000)

        The output of the RNN layer contains a single vector per sample corresponding to the last
        timestep. This vector contains information about the entire input sequence. The output shape
        is (batch_size, units) where 'units' corresponds to the 'units' argument passed to the layer
        constructor.
        The RNN layer can return the entire sequence of outputs of each sample when
        'return_sequences=True'. The shape of the output is: (batch_size, timesteps, units)

        In our case, units=1024 and after the Dense layer of 'vocab_size' size the output of the
        entire model is (1, 21, 15000)
        """
        tokenized_target_sentence = target_vectorization([decoded_sentence])

        """
        Note that this inference setup, while very simple, is rather inefficient, since we reprocess
        the entire source sentence and the entire generated target sentence every time we sample a
        new word. In a practical application, you'd factor the encoder and the decoder as two
        separate models, and your decoder would only run a single step at each token-sampling
        iteration, reusing its previous internal state.
        """
        next_token_predictions = model.predict(
            [tokenized_input_sentence, tokenized_target_sentence]
        )

        """
        The output shape is (batch_size, timesteps, vocab_size) == (1, 21, 15000) so here we take
        the max prob in the vocab_size axis at each time step 'i'.
        """
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

"""
Output:

$ python rnn_predict.py
2022-07-05 11:09:16.438606: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
-
He broke the window intentionally.
1/1 [==============================] - 3s 3s/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
[start] Él fue la ventana de la ventana [end]
-
I'm going to have a baby.
1/1 [==============================] - 0s 45ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 45ms/step
[start] voy a tener un bebé [end]
-
We must go.
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 41ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
[start] tenemos que ir [end]
-
This is a serious matter.
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 41ms/step
[start] este es un problema sin [UNK] [end]
-
He's head over heels in love with Mary.
1/1 [==============================] - 0s 41ms/step
1/1 [==============================] - 0s 41ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 41ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 51ms/step
[start] Él está en la [UNK] de la [UNK] de mary [end]
-
Winter is my favorite season.
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 49ms/step
[start] el verano es mi estación más grande [end]
-
Mary felt like Tom was undressing her with his eyes.
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 51ms/step
[start] mary se estaba como si mary se [UNK] con su perro [end]
-
Their company created forty new jobs.
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 52ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 53ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 52ms/step
[start] su nuevo [UNK] de [UNK] nuevo [end]
-
He didn't study at all.
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 48ms/step
[start] no Él nada de todo [end]
-
Back off!
1/1 [==============================] - 0s 48ms/step
[start] [end]
-
Let me explain.
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 46ms/step
[start] déjame [UNK] [end]
-
He made an apology to us for being late.
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 51ms/step
[start] Él hizo una buena para que nos [UNK] tarde [end]
-
She is as young as I am.
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 46ms/step
[start] ella es tan joven como yo [end]
-
It is not so difficult as you think.
1/1 [==============================] - 0s 45ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 45ms/step
1/1 [==============================] - 0s 47ms/step
1/1 [==============================] - 0s 48ms/step
[start] no es tan difícil como tú [end]
-
I am taking a rest in my car.
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 49ms/step
[start] me estoy [UNK] un auto en mi auto [end]
-
Fish don't like sunlight.
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 51ms/step
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 52ms/step
1/1 [==============================] - 0s 47ms/step
1/1 [==============================] - 0s 49ms/step
1/1 [==============================] - 0s 47ms/step
1/1 [==============================] - 0s 44ms/step
[start] a los no les gustan las [UNK] [end]
-
Nobody's going to hire you.
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 48ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 46ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 46ms/step
[start] nadie te va a [UNK] [end]
-
Open your suitcase.
1/1 [==============================] - 0s 50ms/step
1/1 [==============================] - 0s 45ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 41ms/step
[start] [UNK] su mano [end]
-
I didn't even know you knew.
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 42ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
[start] ni siquiera sabía [end]
-
Who are we kidding?
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 43ms/step
1/1 [==============================] - 0s 43ms/step
[start] con quién estamos [end]
"""