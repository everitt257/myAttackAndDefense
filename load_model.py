import tensorflow as tf
from models import cnn
import os

test_default_path = "./saved_model_weights/cnn_mnist.ckpt"

def loadModel(model, path):
    assert os.path.exists(os.path.split(path)[0])==True, "path not exist"
    assert isinstance(path, str)
    model.load_weights(path)


if __name__ == '__main__':
    model = cnn.basic_model()
    model = loadModel(model, test_default_path)