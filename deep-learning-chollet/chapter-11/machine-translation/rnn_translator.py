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