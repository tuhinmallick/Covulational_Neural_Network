from Layers import *
from Optimization import *
import numpy as np

import copy


class NeuralNetwork:
    def __init__(self, optimizer,weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.wi= weights_initializer
        self.bi= bias_initializer
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        tmp = np.copy(self.input_tensor)

        for i in self.layers:
            op = i.forward(tmp)
            tmp = op

        return self.loss_layer.forward(tmp, self.label_tensor)

    def backward(self):

        tmp = self.loss_layer.backward(self.label_tensor)

        for i in reversed(self.layers):
            op = i.backward(tmp)
            tmp = op

        return tmp

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.wi,self.bi)
        self.layers.append(layer)

    def train(self, iterations):

        for i in range(0, iterations):
            fwd = NeuralNetwork.forward(self)
            backward = NeuralNetwork.backward(self)
            self.loss.append(fwd)
        return np.array(self.loss)

    def test(self, input_tensor):

        tmp = input_tensor
        for i in self.layers:
            op = i.forward(tmp)
            tmp = op

        return tmp