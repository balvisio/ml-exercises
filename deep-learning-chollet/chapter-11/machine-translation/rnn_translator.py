import json
import pickle
from tensorflow import keras
from tensorflow.keras import layers

import translation


def get_model(vocab_size):
    embed_dim = 256
    latent_dim = 1024

    source = keras.Input(shape=(None,), dtype="int64", name="english")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
    encoded_source = layers.Bidirectional(
        layers.GRU(latent_dim), merge_mode="sum")(x)

    past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
    decoder_gru = layers.GRU(latent_dim, return_sequences=True)
    x = decoder_gru(x, initial_state=encoded_source)
    x = layers.Dropout(0.5)(x)
    target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
    seq2seq_rnn = keras.Model([source, past_target], target_next_step)

    return seq2seq_rnn


if __name__ == "__main__":
    vocab_size = 15000
    train_pairs, val_pairs, test_pairs = translation.create_text_pairs()
    source_vectorization, target_vectorization = translation.create_text_vectorization(
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

    train_ds = translation.make_dataset(train_pairs, source_vectorization, target_vectorization)
    val_ds = translation.make_dataset(val_pairs, source_vectorization, target_vectorization)

    model = get_model(vocab_size)

    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"], # In a real-world example, we would use BLEUscore instead of accuracy
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "seq2seq_rnn.keras",
            save_best_only=True,
        )
    ]

    model.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)


"""
Training in the convergencelab vs. local Mac

ConvergenceLab

# python deep-learning-chollet/chapter-11/machine-translation/rnn_translator.py
2022-07-05 08:57:15.278543: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-07-05 08:57:18.307955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30574 MB memory:  -> device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:83:00.0, compute capability: 7.0
2022-07-05 08:57:18.312056: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 30520 MB memory:  -> device: 1, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0
Epoch 1/15
2022-07-05 08:59:10.058872: W tensorflow/core/common_runtime/forward_type_inference.cc:231] Type inference failed. This indicates an invalid graph that escaped type checking. Error message: INVALID_ARGUMENT: expected compatible input types, but input 1:
type_id: TFT_OPTIONAL
args {
  type_id: TFT_PRODUCT
  args {
    type_id: TFT_TENSOR
    args {
      type_id: TFT_LEGACY_VARIANT
    }
  }
}
 is neither a subtype nor a supertype of the combined inputs preceding it:
type_id: TFT_OPTIONAL
args {
  type_id: TFT_PRODUCT
  args {
    type_id: TFT_TENSOR
    args {
      type_id: TFT_INT8
    }
  }
}

	while inferring type of node 'cond_41/output/_24'
2022-07-05 08:59:11.553276: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
1302/1302 [==============================] - 151s 100ms/step - loss: 1.6292 - accuracy: 0.4195 - val_loss: 1.3102 - val_accuracy: 0.5083
Epoch 2/15
1302/1302 [==============================] - 126s 97ms/step - loss: 1.3130 - accuracy: 0.5285 - val_loss: 1.1497 - val_accuracy: 0.5714
Epoch 3/15
1302/1302 [==============================] - 123s 94ms/step - loss: 1.1715 - accuracy: 0.5779 - val_loss: 1.0692 - val_accuracy: 0.6017
Epoch 4/15
1302/1302 [==============================] - 117s 90ms/step - loss: 1.0809 - accuracy: 0.6098 - val_loss: 1.0367 - val_accuracy: 0.6197
Epoch 5/15
1302/1302 [==============================] - 120s 92ms/step - loss: 1.0345 - accuracy: 0.6332 - val_loss: 1.0212 - val_accuracy: 0.6292
Epoch 6/15
1302/1302 [==============================] - 115s 89ms/step - loss: 1.0050 - accuracy: 0.6508 - val_loss: 1.0165 - val_accuracy: 0.6348
Epoch 7/15
1302/1302 [==============================] - 117s 90ms/step - loss: 0.9856 - accuracy: 0.6651 - val_loss: 1.0197 - val_accuracy: 0.6373
Epoch 8/15
1302/1302 [==============================] - 117s 90ms/step - loss: 0.9719 - accuracy: 0.6749 - val_loss: 1.0196 - val_accuracy: 0.6403
Epoch 9/15
1302/1302 [==============================] - 121s 93ms/step - loss: 0.9625 - accuracy: 0.6820 - val_loss: 1.0214 - val_accuracy: 0.6416
Epoch 10/15
1302/1302 [==============================] - 124s 95ms/step - loss: 0.9545 - accuracy: 0.6887 - val_loss: 1.0249 - val_accuracy: 0.6421
Epoch 11/15
1302/1302 [==============================] - 125s 96ms/step - loss: 0.9490 - accuracy: 0.6932 - val_loss: 1.0289 - val_accuracy: 0.6426
Epoch 12/15
1302/1302 [==============================] - 124s 95ms/step - loss: 0.9453 - accuracy: 0.6963 - val_loss: 1.0314 - val_accuracy: 0.6434
Epoch 13/15
1302/1302 [==============================] - 126s 97ms/step - loss: 0.9425 - accuracy: 0.6984 - val_loss: 1.0332 - val_accuracy: 0.6438
Epoch 14/15
1302/1302 [==============================] - 125s 96ms/step - loss: 0.9414 - accuracy: 0.6991 - val_loss: 1.0359 - val_accuracy: 0.6426
Epoch 15/15
1302/1302 [==============================] - 126s 97ms/step - loss: 0.9408 - accuracy: 0.7003 - val_loss: 1.0372 - val_accuracy: 0.6431


Mac

$ python deep-learning-chollet/chapter-11/machine-translation/rnn_translator.py
2022-07-04 19:48:43.205948: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/15
1302/1302 [==============================] - 5199s 4s/step - loss: 1.6404 - accuracy: 0.4150 - val_loss: 1.3210 - val_accuracy: 0.5022
Epoch 2/15
1302/1302 [==============================] - ETA: 0s - loss: 1.3199 - accuracy: 0.5246^CTraceback (most recent call last):
  File "/Users/balvisio/repos/ml-exercises/deep-learning-chollet/chapter-11/machine-translation/rnn_translator.py", line 53, in <module>
    model.fit(train_ds, epoch
"""