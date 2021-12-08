import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.max_pool = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Create a pool shape and initialize it with least values possible.

        if self.pooling_shape == (1, 1):
            max_pool = np.ones_like(input_tensor)
        # elif self.pooling_shape[0] == 1:
        #     max_pool = np.ones_like(input_tensor[::, ::, ::, :-(self.pooling_shape[1] - 1)])
        # elif self.pooling_shape[1] == 1:
        #     max_pool = np.ones_like(input_tensor[::, ::, :-(self.pooling_shape[0] - 1), ::])
        else:
            max_pool = np.ones_like(input_tensor[::, ::, :-(self.pooling_shape[0] - 1), :-(self.pooling_shape[1] - 1)])

        max_pool = max_pool * -np.inf

        # Pick out the maximum values amongst the array and append to max_pool.

        for i in range(self.pooling_shape[0]):
            for j in range(self.pooling_shape[1]):
                max_pool = np.maximum(max_pool, input_tensor[::, ::, i:i+max_pool.shape[2], j:j+max_pool.shape[3]])


        # Consider the stride and create a sub-sampled max_pool variable

        self.max_pool = np.zeros_like(max_pool)
        self.max_pool[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]] = max_pool[::, ::, ::self.stride_shape[0],
                                                                           ::self.stride_shape[1]]

        return self.max_pool[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]]

    def backward(self, error_tensor):
        # resize error_tensor to reverse stride access
        padded_error = np.zeros_like(self.max_pool)
        padded_error[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        gradient_tensor = np.zeros_like(self.input_tensor)

        # - traverse each input tensor with a max pool kernel of size pool_shape
        # - self.input_tensor[::, ::, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]] gives input pixels for [i,j]th max pooling step
        # shape of above data structure is - (batch_size, num_channels, pool_shape[0], pool_shape[1])
        # - self.max_pool[::, ::, i, j] gives max value of the above for each channel in each batch
        # reshape above to - (batch_size, num_channels, 1, 1) so that it can be compared to the input pixels of [i,j]th max pooling step.
        # - comparison gives mask where only the position of max value is marked 1
        # - error_tensor[::, ::, i, j] gives errors w.r.t. [i,j]th max pooling step.
        # reshape above to - (batch_size, num_channels, 1, 1)
        # - multiply it with the mask - this ensures errors are propagated only along the max values.
        # - to consider overlapping cases, add newly computed values to old values of adding gradient_tensor
        for i in range(self.input_tensor.shape[2] - self.pooling_shape[0] + 1):
            for j in range(self.input_tensor.shape[3] - self.pooling_shape[1] + 1):
                gradient_tensor[::, ::, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]] += \
                    (self.max_pool[::, ::, i, j].reshape(*self.input_tensor.shape[:2], 1, 1) == \
                     self.input_tensor[::, ::, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]]) * \
                    padded_error[::, ::, i, j].reshape(*self.input_tensor.shape[:2], 1, 1)

        return gradient_tensor