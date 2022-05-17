import numpy as np
import tensorflow as tf
import spektral

adj, features, labels, train_mask, val_mask, test_mask = spektral.datasets.citation.load_data(dataset_name="cora")

features = features.todense()
adj = adj.todense() + np.eye(adj.shape[0])

features = features.astype("float32")
adj = adj.astype("float32")

print(features.shape)
print(adj.shape)
print(labels.shape)

print(np.sum(train_mask))
print(np.sum(val_mask))
print(np.sum(test_mask))

def masked_softmax_cross_entropy(logits, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
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

def gnn(fts, adj, transform, activation): # node feature matrix, adjacency matrix
    seq_fts = transform(fts)
    ret_fts = tf.matmul(adj, seq_fts)
    return activation(ret_fts)

def train_cora(fts, adj, gnn_fn, units, epochs, lr):
    lyr_1 = tf.keras.layers.Dense(units)
    lyr_2 = tf.keras.layers.Dense(7)

    def core_gnn(fts, adj):
        hidden = gnn_fn(fts, adj, lyr_1, tf.nn.relu)
        logits = gnn_fn(hidden, adj, lyr_2, tf.identity)
        return logits
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    best_accuracy = 0.0

    for ep in range(epochs + 1):
        with tf.GradientTape() as t:
            logits = core_gnn(fts, adj)
            loss = masked_softmax_cross_entropy(logits, labels, train_mask)
        variables = t.watched_variables()
        grads = t.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        logits = core_gnn(fts, adj)

        val_accuracy = masked_accuracy(logits, labels, val_mask)
        test_accuracy = masked_accuracy(logits, labels, test_mask)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"Epoch: {ep} | Training Loss: {loss.numpy()} | Val accuracy: {val_accuracy} | Test Accuracy: {test_accuracy.numpy()}")

#train_cora(features, adj, gnn, 32, 200, 0.01)

# In this case we don't use the graph at all (identity matrix). The performance is poorer than
# with the graph. This shows that the graph is useful.
# train_cora(features, tf.eye(adj.shape[0]), gnn, 32, 200, 0.01)

deg = tf.reduce_sum(adj, axis=-1)
#train_cora(features , adj / deg, gnn, 32, 200, 0.01)

norm_deg = tf.linalg.diag(1.0 / tf.sqrt(deg))
norm_adj = tf.matmul(norm_deg, tf.matmul(adj, norm_deg))
train_cora(features, norm_adj, gnn, 32, 200, 0.01)