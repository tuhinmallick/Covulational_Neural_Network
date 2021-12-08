from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        shifted_ip_tensor = input_tensor - np.max(input_tensor)
        exp_ip_tensor = np.exp(shifted_ip_tensor)
        self.output_tensor = exp_ip_tensor / np.sum(exp_ip_tensor, axis=1, keepdims=True)

        return self.output_tensor

    def backward(self, error_tensor):
        return self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))

# class SoftMax(BaseLayer):      #different version of code,however with for loop, affects speed.
#
#     def __init__(self):
#         super().__init__()
#         self.forw=None
#
#     def forward(self, input_tensor):
#         # exponents = np.exp(input_tensor[i] - np.max(input_tensor[i]))
#         exponents = np.concatenate([np.expand_dims(np.exp(input_tensor[i] - np.max(input_tensor[i])), axis=0) for i in range(input_tensor.shape[0])])
#
#         self.forw=np.concatenate([np.expand_dims(exponents[i]/np.sum(exponents[i]),axis=0) for i in range(input_tensor.shape[0])])
#
#         return self.forw
#
#     def backward(self, error_tensor):
#
#
#         back_soft = np.concatenate([np.expand_dims(error_tensor[i]-np.sum(np.multiply(error_tensor[i], self.forw[i])), axis=0) for i in range(error_tensor.shape[0])])
#         input_error = np.multiply(self.forw, back_soft)
#
#         return input_error