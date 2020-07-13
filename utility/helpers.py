import tensorflow as tf

def guess_top_label(model, x):
    logits = model(x)
    return tf.argmax(tf.nn.softmax(logits,axis=1), axis=1)