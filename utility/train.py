import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Prepare the training metrics
train_acc_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
val_acc_metric = tf.keras.metrics.SparseCategoricalCrossentropy()

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return y_, loss_object(y_true=y, y_pred=y_)

@tf.function
def train_step(model, x_batch, y_batch, training):
    with tf.GradientTape() as tape:
        y_, loss_value = loss(model, x_batch, y_batch, training)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch, y_)
    return loss_value

@tf.function
def test_step(model, x_batch, y_batch):
    val_logits = model(x_batch, training=False)
    val_acc_metric.update_state(y_batch, val_logits)