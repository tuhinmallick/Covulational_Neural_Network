import numpy as np


class Constant:
    def __init__(self, cv=0.1):
        self.cv = cv            #determines the constant value used for weight initialization.

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.cv)#returns an initialized tensor of the desired shape.


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random(weights_shape)#returns an initialized tensor of the desired shape.


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sd = np.sqrt(2 / (fan_out + fan_in))

        return np.random.normal(0, sd, weights_shape)#returns an initialized tensor of the desired shape.


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sd = np.sqrt(2 / fan_in)
        return np.random.normal(0, sd, weights_shape)#returns an initialized tensor of the desired shape.