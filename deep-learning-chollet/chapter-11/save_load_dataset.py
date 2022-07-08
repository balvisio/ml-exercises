import os, pathlib, shutil, random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

    batch_size = 1
    
    test_ds = keras.utils.text_dataset_from_directory(
        base_dir / "test", batch_size=batch_size, shuffle=True
    )
    return test_ds


def create_int_ds(test_ds):
    max_sequence_length = 1
    max_tokens = 20000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_sequence_length,
    )
    
    text_only_train_ds = test_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)
    int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

    return int_test_ds

test_ds = create_ds()
int_test_ds = create_int_ds(test_ds)

take_test_ds = int_test_ds.take(1)

# Changes from for loop to for loop because shuffle=True
for e in take_test_ds:
    print(e)
for e in take_test_ds:
    print(e)
for e in take_test_ds:
    print(e)
for e in take_test_ds:
    print(e)

print("Saving...")
tf.data.experimental.save(take_test_ds, "int_test_ds") # Save can save any of the samples

print("Loading...")
new_take_test_ds = tf.data.experimental.load("int_test_ds")

# Won't change from for loop to for loop because it is the sampled saved
for e in new_take_test_ds:
    print(e)
for e in new_take_test_ds:
    print(e)
for e in new_take_test_ds:
    print(e)
for e in new_take_test_ds:
    print(e)