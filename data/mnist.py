import tensorflow as tf

class MNIST(object):
    def __init__(self):
        self.train, self.test = tf.keras.datasets.mnist.load_data()
        self.makeDataset()
    
    @staticmethod
    def normalize_expand_img(image, label):
        """Normalizes images and expand one dimension: `uint8` -> `float32`."""
        image = tf.expand_dims(image, -1)
        return tf.cast(image, tf.float32) / 255., label

    def makeDataset(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train)
        test_dataset = tf.data.Dataset.from_tensor_slices(self.test)

        train_dataset = train_dataset.map(MNIST.normalize_expand_img)
        test_dataset = test_dataset.map(MNIST.normalize_expand_img)

        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 100

        self.train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.test_dataset = test_dataset.batch(BATCH_SIZE)