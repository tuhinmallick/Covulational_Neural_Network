from Layers.Base import BaseLayer
import numpy as np



class FullyConnected(BaseLayer):


    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.ip = input_size
        self.op = output_size
        self.trainable = True
        self.bias= None
        self.optimizer = None
        self.gradient_weights = None

        self.weights = np.random.rand(input_size + 1, output_size)


    def forward(self, input_tensor):

        self.input_tensor = np.concatenate((input_tensor, np.transpose(np.array([np.ones(len(input_tensor))]))), axis=1)

        foutput = np.dot(self.input_tensor, self.weights)

        return foutput

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def backward(self, error_tensor):
        self.input_error = np.dot(error_tensor, self.weights[:-1].T)

        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)


        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return np.copy(self.input_error)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad):
        self._gradient_weights = grad

    def initialize(self,weights_initializer,bias_initializer):
        self.weights[:-1] = weights_initializer.initialize(self.weights[:-1].shape, self.weights[:-1].shape[0],
                                                           self.weights[:-1].shape[1])

        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, 1, self.weights[-1].shape[0])
