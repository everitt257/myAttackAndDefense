import tensorflow as tf
from utility.helpers import guess_top_label
"""
The base class for different attacks
"""
class Attack(object):
    def __init__(self, model):
        self.model = model

    def generate(self, x, **kwargs):
        """
        returns perturbed signal that can be added to x
        """
        msg = "sub class must implement this"
        raise NotImplementedError(msg)
        return x
    
    def generate_adversarial_sample(self, x, **kwargs):
        perturbations = self.generate(x, **kwargs)
        msg = "sub class must implement this"
        raise NotImplementedError(msg)
        return x+perturbations

    def evaluate(self, x, **kwargs):
        self.metric = tf.metrics.Accuracy()
        y_true = guess_top_label(self.model, x)
        y_pred = guess_top_label(self.model, self.generate_adversarial_sample(x, **kwargs))
        self.metric.update_state(y_pred=y_pred, y_true=y_true)
        result = self.metric.result().numpy()
        self.metric.reset_states()
        return result
