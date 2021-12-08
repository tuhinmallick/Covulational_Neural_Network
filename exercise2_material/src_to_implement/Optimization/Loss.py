import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor=None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor= input_tensor
        return -np.sum(np.multiply(label_tensor, np.log(input_tensor + np.finfo(float).eps)))

    def backward(self, label_tensor):
        return -label_tensor / self.input_tensor