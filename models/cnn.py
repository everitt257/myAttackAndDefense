import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
This script builds a basic cnn model with keras sequential api
"""
def basic_model():
    num_classes = 10
    filters = [64, 128, 128]
    kernel_sizes = [8, 6, 5]
    strides = [2, 2, 1]
    paddings = ['same', 'valid', 'valid']
    activation = 'relu'


    model = keras.Sequential(name='basic_cnn')
    model.add(layers.Input(shape=(28, 28, 1), name='input'))

    for (filter, kernel_size, stride, padding) in zip(filters, kernel_sizes, strides, paddings):
        layer = layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=stride, padding=padding, activation=activation)
        model.add(layer)

    model.add(layers.Flatten())
    model.add(layers.Dense(units=num_classes, activation=None))

    return model
    