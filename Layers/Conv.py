import copy

import numpy as np
from scipy.signal import convolve, correlate
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    '''While fully connected layers are theoretically well suited to approximate any function they struggle to efficiently classify images due to extensive memory consumption and overfitting. Using convolutional layers, these problems can be circumvented by restricting the layer's pa- rameters to local receptive fields.'''
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True   #this layer has trainable parameters,

        self.stride_shape = stride_shape
        '''convolution_shape : determines whether this object provides a 1D or a 2D con- volution layer. For 1D, it has the shape [c, m], whereas for 2D, it has the shape [c, m, n], where c represents the number of input channels, and m, n represent the spatial extent of the filter kernel.'''
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.num_channels = convolution_shape[0]

        '''return the gradient with respect to the weights and bias, after they have been calculated in the backward-pass.'''
        self.weights = np.random.uniform(0.0, 1.0, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0.0, 1.0, num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None

        self.optimizer = None
        self.input_tensor = None

    def forward(self, input_tensor):
        '''returns a tensor that serves as the input tensor for the next layer.'''
        self.input_tensor = input_tensor    #Input size is X x y x S
        batch_size = self.input_tensor.shape[0]        # X
        # Creating output shape
        output_tensor = np.empty((batch_size, self.num_kernels, *input_tensor.shape[2:]))  

        # Correlation between each image and the total kernels involved.
        '''The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order. Here, b stands for the batch, c represents the channels and x, y represent the spatial dimensions.'''
        for b, sample in enumerate(input_tensor):
            for k, kernel in enumerate(self.weights):
                corr = correlate(sample, kernel, mode='same')[self.num_channels // 2] #The output is the same size as sample, centered with respect to the ‘full’ output.
                output_tensor[b][k] = corr + self.bias[k]

        # Case for 1-D, the last dimension avoided during striding.
        #striding helps incoporating pooling mechanism and the dimensionality reduction mechanism into the convolution
        """
        It’s like skipping one step at each point. 
        So, with the stride s, we describe an offset and then we intrinsically produce an activation map that has a lower dimension that is dependent on this stride.
        we reduce the size of the output by a factor of s because skipping so many steps and mathematically this is simply convolution and subsampling at the same time.
        """
        if len(self.convolution_shape) == 2:  # 1-D tensor
            strided_output = output_tensor[::, ::, ::self.stride_shape[0]]
        else:   #2-D tensor
            strided_output = output_tensor[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]]

        return strided_output

    def backward(self, error_tensor):
        # Initializing gradient objects to zero.

        gradient_weights = np.zeros(self.weights.shape)
        gradient_bias = np.zeros(self.bias.shape)

        # Up-sampling to match sizes.

        upsampled_error = np.zeros((*error_tensor.shape[0:2], *self.input_tensor.shape[2:]))

        # Case for 1-D

        if len(self.convolution_shape) == 2:  # 1-D tensor
            upsampled_error[::, ::, ::self.stride_shape[0]] = error_tensor
        else:
            upsampled_error[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor

        # Flipping Kernels.

        rearranged_weights = np.swapaxes(self.weights, 0, 1)        #Interchange two axes of an array.
        rearranged_weights = np.flip(rearranged_weights, axis=1)       #Reverse the order of elements in an array along columns.

        # Calculating the Gradient wrt the Input.

        output_tensor = np.zeros_like(self.input_tensor)
        for b, error in enumerate(upsampled_error):
            for c, channel in enumerate(rearranged_weights):
                conv = convolve(error, channel, mode='same')        #Fourier transform to actually perform the convolution
                output_tensor[b][c] = conv[self.num_kernels // 2]

        # Padding the input for calculating gradient wrt weights.
        #Padding solves the problem of missing observations at the boundary
        if len(self.convolution_shape) == 2:
            padded_input = self.input_tensor
        else:
            padded_input = np.pad(self.input_tensor, (
                (0, 0), (0, 0), ((self.convolution_shape[1] - 1) // 2, self.convolution_shape[1] // 2),
                ((self.convolution_shape[2] - 1) // 2, self.convolution_shape[2] // 2)), 'constant')    #Pads input_tensor with a constant value

        # Calculating gradient wrt weights and bias.

        for b, error in enumerate(upsampled_error):
            for k, err_channel in enumerate(error):
                for c in range(self.num_channels):
                    gradient_weights[k][c] += correlate(padded_input[b][c], err_channel, mode='valid')
                gradient_bias[k] += np.sum(err_channel)

        self.gradient_weights = gradient_weights
        self.gradient_bias = gradient_bias

        # Updating the values for weights and bias required in the next pass

        if self.optimizer:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):

        # fan_in = ip_channels * kernel_spatial_dim, fan_out = op_channels *kernel_spatial_dim

        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.weights.shape[1:]),
                                                      self.weights.shape[0] * np.prod(self.weights.shape[2:]))
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.bias.shape), np.prod(self.bias.shape))

    @property
    def optimizer(self):
        return self._weights_optimizer is not None and self._weights_optimizer is not None

    @optimizer.setter
    def optimizer(self, value):
        self._weights_optimizer = copy.deepcopy(value)
        self._bias_optimizer = copy.deepcopy(value)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value
