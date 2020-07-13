import tensorflow as tf

class MNIST(object):
    def __init__(self, BATCH_SIZE=64, SHUFFLE_BUFFER_SIZE=100):
        self.train, self.test = tf.keras.datasets.mnist.load_data()
        self.batch_size = BATCH_SIZE
        self.shuffle_buffer_size = SHUFFLE_BUFFER_SIZE
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

        self.train_dataset = train_dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size)
        self.test_dataset = test_dataset.batch(self.batch_size)

if __name__ == "__main__":
    mnist = MNIST()
    train_dataset, test_dataset = mnist.train_dataset, mnist.test_dataset
    for test_batch in test_dataset.take(1):
        pass
    x = test_batch[0][:3]