import tensorflow as tf
from models import cnn

test_default_path = "./saved_model_weights/cnn_mnist.ckpt"

def loadModel(model, path):
    assert isinstance(path, str)
    return model.load_weights(path)


model = cnn.basic_model()
model = loadModel(model, test_default_path)