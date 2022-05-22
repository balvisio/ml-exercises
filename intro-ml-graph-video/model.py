"""
Standrard Cora Benchmark:
- Citation network to classify papers into topics
- 2708 nodes, 5429 edges
- Bag of words features: 7 topics
- 140 training, 500 validation, 100 test

This codes implements:
- sum-pooling
- mean-pooling
- GCN

Link to video: https://www.youtube.com/watch?v=8owQBFAHw7E
"""

import numpy as np
import tensorflow as tf
import spektral

# train_mask, val_mask, test_mask tell us which nodes belong to each part of the dataset: training, validation, test, respectively.
adj, features, labels, train_mask, val_mask, test_mask = spektral.datasets.citation.load_data(dataset_name="cora")

features = features.todense() # Convert sparse to dense representation
adj = adj.todense() + np.eye(adj.shape[0])

features = features.astype("float32")
adj = adj.astype("float32")

print(f"Features shape: {features.shape}")
print(f"Adjacency matrix: {adj.shape}")
print(f"Labels: {labels.shape}")

print(f"Training samples: {np.sum(train_mask)}")
print(f"Validation samples: {np.sum(val_mask)}")
print(f"Test samples: {np.sum(test_mask)}")

print(f"Training mask shape: {train_mask.shape}") # shape = (number of nodes,)
print(f"Test mask shape: {test_mask.shape}")

def masked_softmax_cross_entropy(logits, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #Note: loss.shape = (number of nodes,)
    mask = tf.cast(mask, dtype=tf.float32)
    # We divide by the mean so if, for example, we have loss = [x; y; z; n; p]
    # mask = [ 1; 1; 1; 0; 0]
    # => mask / reduce_mean = [ 5/3; 5/3; 5/3; 0; 0 ]
    # We do this to account that at the end we take the "reduce_mean" again but the vector
    # length is of all nodes and not just the train, val or test.
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(logits, labels, mask):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype = tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

# This is the graph neural network layer
def gnn(fts, adj, transform, activation): # node feature matrix, adjacency matrix
    seq_fts = transform(fts) # The transform for this example is the matrix "W". This is the equivalent of the matrix "W" in slides
    ret_fts = tf.matmul(adj, seq_fts)
    return activation(ret_fts)

# We create a 2-layer GNN
# 'units' param is the number of dimensions in the latent space
def train_cora(fts, adj, gnn_fn, units, epochs, lr):
    lyr_1 = tf.keras.layers.Dense(units) # Matrix W from slides
    lyr_2 = tf.keras.layers.Dense(7)

    def cora_gnn(fts, adj):
        hidden = gnn_fn(fts, adj, lyr_1, tf.nn.relu)
        logits = gnn_fn(hidden, adj, lyr_2, tf.identity)
        return logits
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    best_accuracy = 0.0

    for ep in range(epochs + 1):
        with tf.GradientTape() as t:
            logits = cora_gnn(fts, adj)
            loss = masked_softmax_cross_entropy(logits, labels, train_mask)
        variables = t.watched_variables()
        """
        In this network we have 4 watched variables:
          dense/kernel:0 - (1433, 42)
          dense/bias:0 - (42,)
          dense_1/kernel:0 - (42, 7)
          dense_1/bias:0 - (7,)
        
          The first pair comes from the first Dense layer: they a 'W' and 'b'
          Idem for the second pair: come from the second dense layer
        """
        grads = t.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        logits = cora_gnn(fts, adj)

        val_accuracy = masked_accuracy(logits, labels, val_mask)
        test_accuracy = masked_accuracy(logits, labels, test_mask)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"Epoch: {ep} | Training Loss: {loss.numpy()} | Val accuracy: {val_accuracy} | Test Accuracy: {test_accuracy.numpy()}")

# Case 1: Test drive
#train_cora(features, adj, gnn, 32, 200, 0.01)

# Case 2: In this case we don't use the graph at all (identity matrix). The performance is poorer than
# with the graph. This shows that the graph is useful.
# train_cora(features, tf.eye(adj.shape[0]), gnn, 32, 200, 0.01)

# Case 3: This is mean-pooling. 'deg' is the degree matrix of shape (number of nodes,)
deg = tf.reduce_sum(adj, axis=-1)
#train_cora(features , adj / deg, gnn, 32, 200, 0.01)

# Case 4: GCN model.
norm_deg = tf.linalg.diag(1.0 / tf.sqrt(deg))
# norm_adj == D^(-0.5) * A * D^(-0.5) # See Graph ML General Notes
norm_adj = tf.matmul(norm_deg, tf.matmul(adj, norm_deg))
train_cora(features, norm_adj, gnn, 32, 200, 0.01)
