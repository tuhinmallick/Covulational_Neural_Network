from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    """
    ReLu sets the negative half space to Zero and positive half space to x.
    Hence, derivative to be 1 for the entire positive half space and 0 everywhere else.
    Pros: Good generalisation, peicewise linearity helps in speeding up as we dont need exp func.
    No vanishing gradients because there is a large areas of high value for the derivative of this func.
    No unsupervised pre-training
    Implementation is easy, because first derivative is 1 if the area is positive, but 0 everywhere else.
    No second order derivative
    Cons: Not zero centered
    Dying ReLus because it's first derivative is 0 in negative space. Mainly happens due to high learning rate. Solution is Leaky ReLu
    Helps in building deeper networks i.e which have more than 3 hidden layers
    """

    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        tmp = np.maximum(0, self.input_tensor) #f=Max(0,)

        return tmp

    def backward(self, error_tensor):
        tmp1 = np.copy(error_tensor)
        tmp1[self.input_tensor <= 0] = 0

        return tmp1