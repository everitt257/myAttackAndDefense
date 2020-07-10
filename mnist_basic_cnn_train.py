from data import mnist
from models import cnn
import tensorflow as tf

mnist = mnist.MNIST()
train_dataset, test_dataset = mnist.train_dataset, mnist.test_dataset

cnn_model = cnn.basic_model()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def train_step(model, x_batch, y_batch, training):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x_batch, y_batch, training)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))


