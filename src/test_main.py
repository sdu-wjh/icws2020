from test import DataLoader
from test2 import RNN
import tensorflow as tf

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=50, seq_length=40)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for batch_index in range(1000):
    X, y = data_loader.get_batch(40, 50)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss" % (batch_index))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))