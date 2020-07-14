from attacks.attack import Attack
import tensorflow as tf
from utility.helpers import guess_top_label

class fastgradientattack(Attack):
    def __init__(self, model):
        super(fastgradientattack, self).__init__(model)
    
    
    def generate(self, x, **kwargs):
        """
        x: The image/image batch passed in, dimension should be [batch, h,w,c]
        kwargs: The dictionary containing keyworks regarding the type of attack
        should be contain the following:
        {
            "untargeted": boolean, default true, if false, then keywork 'y_target' should be passed in
            "y_target": depending on the dataset, this should be one-hot-encoded
            "norm": should be either l1/l2/infinite norm
            "epi": should be a very small number, such as 0.01, defaults to 0.01
        }
        """
        shape = list(x.shape)
        # Check if x should be expanded in the zeroth dimension
        if len(shape) == 3:
            x = tf.expand_dims(x, axis=0)

        grads = tf.zeros_like(x)
        if not kwargs or ('untargeted' not in kwargs) or ('untargeted' in kwargs and kwargs['untargeted']==True):
            ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            with tf.GradientTape() as tape:
                tape.watch(x)
                logits = self.model(x)
                labels = tf.argmax(tf.nn.softmax(logits,axis=1), axis=1)
                loss_value = ce_loss(y_true=labels, y_pred=logits)
            grads = tape.gradient(loss_value,  x)
        
        if 'epi' in kwargs and isinstance(kwargs['epi'], float):
            epi = kwargs['epi']
        else:
            epi = 0.01
        return epi*tf.sign(grads)
        # return shape

    def generate_adversarial_sample(self, x, **kwargs):
        perturbations = self.generate(x, **kwargs)
        return perturbations, tf.clip_by_value((x+perturbations), clip_value_min=-1, clip_value_max=1)

    # def evaluate(self, x, **kwargs):
    #     return self.evaluate(x, **kwargs)

if __name__ == "__main__":
    from models.cnn import basic_model
    from data.mnist import MNIST
    model = basic_model()
    mnist = MNIST()
    train_dataset, test_dataset = mnist.train_dataset, mnist.test_dataset
    for test_batch in test_dataset.take(1):
        pass
    x = test_batch[0]
    model.load_weights("./saved_model_weights/cnn_mnist.ckpt")
    fgsm = fastgradientattack(model)
    perturbations, adv_x = fgsm.generate_adversarial_sample(x, untargeted=True, epi=0.05)
    from utility.helpers import guess_top_label
    adv_x_labels = guess_top_label(model, adv_x)
    x_labels = guess_top_label(model, x)
    print(adv_x_labels)
    print(x_labels)
    softmax_adv = tf.nn.softmax(model(adv_x), axis=1)
    softmax_ori = tf.nn.softmax(model(x), axis=1)

    accu = tf.metrics.Accuracy()
    accu.update_state(y_pred=adv_x_labels, y_true=x_labels)
    print(accu.result().numpy())
    accu.reset_states()