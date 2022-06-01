"""
Link to Youtube video: https://www.youtube.com/watch?v=8Xt5fnLnduQ
Code: https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/qm9_ecc.py
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, ops, GlobalSumPool
from spektral.utils import batch_iterator, label_to_one_hot, numpy_to_disjoint # For reference: https://graphneural.network/data-modes/

learning_rate = 1e-3
epochs = 100
batch_size = 32

A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num', #nf = node features
                           ef_keys='type',
                           self_loops=False,
                           auto_pad=False,
                           amount=1000)
y = y[["cv"]].values

# 'X' contains the node features for each module. In this case, we chose the atomic number. (e.g. C = 6)
# X.shape == (1000,)
# Each element of 'X' is a list that contains the atoms (atomic_num) that make up the molecule
# X[999] == [[6][7][6][7][6][6][6]]
# X[999][0] == [6]
# type(X[999][0]) == ndarray

# Get all unique atoms and edges accross all molecules so that we can do one host encoding
# We need to know all unique values to do one hot encoding
# type(X_uniq) = ndarray
# X_uniq = [6 7 8 9]
X_uniq = np.unique([v for x in X for v in np.unique(x)])
E_uniq = np.unique([v for e in E for v in np.unique(e)])
X_uniq = X_uniq[X_uniq != 0]
E_uniq = E_uniq[E_uniq != 0]

X = [label_to_one_hot(x, labels=X_uniq) for x in X]
E = [label_to_one_hot(e, labels=E_uniq) for e in E]

F = X[0].shape[-1] # Dimension of node features
S = E[0].shape[-1] # Dimension fo edge featues
# n_out == 1
n_out = y.shape[-1] # Dimension fo the targets
A_train, A_test, X_train, X_test, E_train, E_test, y_train, y_test = train_test_split(A, X, E, y, test_size=0.1, random_state=42)

# Build model
X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True, name="A_in")
E_in = Input(shape=(S,), name="E_in")
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

X_1 = EdgeConditionedConv(64, activation="relu")([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(64, activation="relu")([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out)(X_3)

model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = MeanSquaredError()

@tf.function(
    input_signature=(tf.TensorSpec((None, F), dtype=tf.float64),
        tf.SparseTensorSpec((None, None), dtype=tf.float64),
        tf.TensorSpec((None, S), dtype=tf.float64),
        tf.TensorSpec((None,), dtype=tf.int32),
        tf.TensorSpec((None, n_out), dtype=tf.float64)),
    experimental_relax_shapes=True)
def train_step(X_, A_, E_, I_, y_):
    with tf.GradientTape() as tape:
        predictions = model([X_, A_, E_, I_], training=True)
        loss = loss_fn(y_, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Fit model
current_batch = 0
model_loss = 0
batches_in_epoch = np.ceil(len(A_train) / batch_size)
print("Fitting model")
batches_train = batch_iterator([X_train, A_train, E_train, y_train], batch_size=32, epochs=epochs)

for b in batches_train:
    # b is a list of len 4
    # b[:-1] are all elements of b except the last one
    # *b[:1] "unpacks" the list and makes them 3 variables.
    # The signature of the func: numpy_to_disjoint(X_list, A_list, E_list=None). Thus, by using the
    # asterisk it converts the list to fn(a, b, c). In other words:
    #
    # fn(*x) == fn(x[0], x[1], x[2])

    X_, A_, E_, I_ = numpy_to_disjoint(*b[:-1])
    # I_ is only needed for pooling, it tells you which nodes belong to which sub-graph.
    # I_ is a vector of shape (None,) (usually around (200,)). It contains the indices of the molecule
    # the row belongs to. E.g. I_ = [0, 0, 1, ..., 1, 2, ..., 2, ..., 31] <= This for a batch of size 32
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[-1]
    outs = train_step(X_, A_, E_, I_, y_)

    model_loss += outs.numpy()
    current_batch += 1
    if current_batch == batches_in_epoch:
        print(f"Loss: {model_loss / batches_in_epoch}")
        model_loss = 0
        current_batch = 0

# Evaluate model
print("Testing model")
model_loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)
batches_test = batch_iterator([X_test, A_test, E_test, y_test], batch_size=batch_size)
for b in batches_test:
    X_, A_, E_, I_ = numpy_to_disjoint(*b[:-1])
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[3]

    predictions = model([X_, A_, E_, I_], training=False)
    model_loss += loss_fn(y_, predictions)

    for yi, pred in zip(y_, predictions):
        print(f"{yi} - {pred}")

model_loss /= batches_in_epoch
print(f"Done. Test loss: {model_loss}")