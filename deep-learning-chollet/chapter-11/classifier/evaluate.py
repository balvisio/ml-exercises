"""
The following code is based on the the book "Deep Learning with Python" by Francois Chollet (2nd
edition), Chapter 11, Section 4.

The code has been annotatated and modified to be more clear and more flexible.
"""
import sys

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from transformer.transformer_encoder import TransformerEncoder

int_test_ds = tf.data.experimental.load("int_test_ds")

model = keras.models.load_model(
    "transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder})

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

