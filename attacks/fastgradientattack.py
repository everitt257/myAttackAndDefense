from attacks.attackbase import Attack
import tensorflow as tf

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
        }
        """

        # Untargeted attack
        grads = tf.zeros_like(x)
        if not kwargs or ('untargeted' in kwargs and kwargs['untargeted']==False):
            ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            with tf.GradientTape as tape:
                logits = self.model(x)
                labels = tf.argmax(tf.nn.softmax(logits,axis=1), axis=1)
                loss_value = ce_loss(y_true=labels, y_pred=logits)
            grads = tape.gradient(loss_value,  x)
        
        return grads

