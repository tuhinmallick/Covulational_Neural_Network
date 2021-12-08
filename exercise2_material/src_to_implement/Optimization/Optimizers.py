import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        new_weight = weight_tensor - (self.lr * gradient_tensor)

        return new_weight


class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate):
        self.lr = learning_rate
        self.mr = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        vk = (self.mr * self.vk) - (self.lr * gradient_tensor)
        new_weight = weight_tensor + vk
        self.vk = vk

        return new_weight


class Adam:
    def __init__(self, learning_rate: float, mu, rho):
        self.iter = 1.
        self.lr = learning_rate
        self.mu = mu
        self.rho = rho

        self.vk = 0
        self.rk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.vk = (self.mu * self.vk) + (1. - self.mu) * gk
        vk_bias = self.vk / (1. - np.power(self.mu, self.iter))
        self.rk = (self.rho * self.rk) + (1. - self.rho) * gk * gk
        rk_bias = self.rk / (1. - np.power(self.rho, self.iter))

        self.iter += 1

        new_weight = weight_tensor - (self.lr * (vk_bias / (np.sqrt(rk_bias) + np.finfo(float).eps)))

        return new_weight
