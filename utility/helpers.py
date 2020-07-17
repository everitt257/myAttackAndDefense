import tensorflow as tf

def guess_top_label(model, x):
    """
    Computes the top label given model and inputs x.
    model: tensorflow model
    x: inputs
    """
    logits = model(x)
    return tf.argmax(tf.nn.softmax(logits,axis=1), axis=1)

def norm(x, ord=2):
    """
    Compute the norm of x given ord.
    x: batch of tensor
    ord: 0,1,2 or np.inf
    """

    x = tf.reshape(x, [x.get_shape()[0], -1])
    return tf.reshape(tf.norm(x, ord=ord, axis=1, keepdims=True), [x.get_shape()[0], -1, 1, 1])