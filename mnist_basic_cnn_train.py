from data import mnist
from models import cnn
import tensorflow as tf
import argparse

# Create parser for command-line interface
my_parser = argparse.ArgumentParser(description="basic model training & saving")
my_parser.add_argument('--save', type=bool, default=False, help="choose whether to save")
my_parser.add_argument('--epoch', type=int, default=10)
my_parser.add_argument('--attack', type=str, default='None')

args = my_parser.parse_args()
 
mnist = mnist.MNIST()
train_dataset, test_dataset = mnist.train_dataset, mnist.test_dataset

model = cnn.basic_model()

# Define loss object/function & optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Prepare the training metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
epoch_loss_avg = tf.keras.metrics.Mean()

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return y_, loss_object(y_true=y, y_pred=y_)

@tf.function
def train_step(model, x_batch, y_batch, training):
    with tf.GradientTape() as tape:
        y_, loss_value = loss(model, x_batch, y_batch, training)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch, model(x_batch, training=True))
    epoch_loss_avg.update_state(loss_value)
    return loss_value

@tf.function
def test_step(model, x_batch, y_batch):
    val_logits = model(x_batch, training=False)
    val_acc_metric.update_state(y_batch, val_logits)


log_train_acc = []
log_val_acc = []
log_train_loss = []

import time

epochs = args.epoch
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch, ))
    start_time = time.time()

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(model, x_batch_train, y_batch_train, training=True)

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("Seen so far: %d samples" % ((step + 1) * mnist.batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    epoch_ave_loss = epoch_loss_avg.result()

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Validation
    for x_batch_val, y_batch_val in test_dataset:
        test_step(model, x_batch_val, y_batch_val)
    
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    log_train_acc.append(train_acc)
    log_val_acc.append(val_acc)
    log_train_loss.append(epoch_ave_loss)

# Todo: write command line interface with argparse

if args.save:
    model.save_weights("./saved_model_weights/cnn_mnist.ckpt")