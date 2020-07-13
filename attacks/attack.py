"""
The base class for different attacks
"""
class Attack(object):
    def __init__(self, model):
        self.model = model

    def generate(self, x, **kwargs):
        """
        returns perturbed image that formulate an attack
        """
        # msg = "sub class must implement this"
        # raise NotImplementedError(msg)
        # return x
        pass
