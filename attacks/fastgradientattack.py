from attacks.attack import Attack
import tensorflow as tf
import numpy as np
from utility.helpers import guess_top_label, norm

class fastgradientattack(Attack):
    def __init__(self, model, eps):
        """
        :model: model specifying the trained model that used to classify
        :eps: float scalar specifying size of constraint region
        """
        super(fastgradientattack, self).__init__(model)
        

    def generate(self, x, **kwargs):
        """
        :x: The image/image batch passed in, dimension should be [batch, h,w,c]
        :kwargs: The dictionary containing keyworks regarding the type of attack
        should be contain the following:
        {
            "untargeted": boolean, default true, if false, then keywork 'y_target' should be passed in
            "y_target": depending on the dataset, this should be one-hot-encoded
            "norm": should be either l1/l2/infinite norm
            "epi": float scalar specifying size of constraint region. Such as 0.01, defaults to 0.01
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
            epi = tf.constant(float(kwargs['epi']))

        if 'norm' in kwargs:
            norm = kwargs['norm']

        optimized_grads = self.optimize_by_norm(grads, norm=norm, epi=epi)

        return optimized_grads
    
    def optimize_by_norm(self, grads, norm=np.inf, epi=tf.constant(0.01)):
        """
        return gradients within boundaries within norm and epi range
        :grads: gradients computed via backpropagation
        :norm: either it be np.inf, 1, 2
        :epi: range of constraints

        TODO: I'm not quite sure how the 1-norm optimization is implemented in
        the cleverhans library. So for the 1-norm, this is a direct copy from
        cleverhans.
        """
        avoid_zero_div = 1e-12
        if norm == np.inf:
            optimized_grads = tf.sign(grads)
        elif norm == 1:
            pass
        elif norm == 2:
            grads_norms = norm(grads, ord=2)
            optimized_grads = grads/tf.maximum(avoid_zero_div, grads_norms) 
        else:
            raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                    "currently implemented.")
        return epi*optimized_grads

    def generate_adversarial_sample(self, x, **kwargs):
        perturbations = self.generate(x, **kwargs)
        return perturbations, tf.clip_by_value((x+perturbations), clip_value_min=-1, clip_value_max=1)


if __name__ == "__main__":
    from models.cnn import basic_model
    from data.mnist import MNIST
    from utility.helpers import guess_top_label
    # load model
    model = basic_model()
    model.load_weights("./saved_model_weights/cnn_mnist.ckpt")
    # load dataset
    mnist = MNIST()
    train_dataset, test_dataset = mnist.train_dataset, mnist.test_dataset
    # specify attack
    fgsm = fastgradientattack(model)
    for test_batch in test_dataset.take(10):
        x = test_batch[0]
        print(fgsm.evaluate(x, untargeted=True, epi=0.01, reset=False))
    
    print("The running average accuracy of 10 batches is:", fgsm.metric.result().numpy())

    