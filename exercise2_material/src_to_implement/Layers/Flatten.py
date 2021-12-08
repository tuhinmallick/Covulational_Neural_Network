import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    '''Flatten layers reshapes the multi-dimensional input to a one dimensional feature vector. This
    is useful especially when connecting a convolutional or pooling layer with a fully connected
    layer.'''
    def __init__(self):
        super(Flatten, self).__init__()
        self.tens_shape = None

    def forward(self, input_tensor):
        self.tensor_shape = input_tensor.shape

        return np.reshape(input_tensor, (input_tensor.shape[0], np.prod(input_tensor.shape[1:]))) 
        #np.prod: Return the product of array elements over a given axis.

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.tensor_shape)