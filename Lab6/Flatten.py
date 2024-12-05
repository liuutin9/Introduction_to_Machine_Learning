import numpy as np

class Flatten():
    def __init__(self):
        pass
    def forward(self, A_prev):
        """
        Implements the forward pass of the flatten layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the flatten layer, a 2-dimensional array of shape (m, (n_H_prev * n_W_prev * n_C_prev))
        """

        # Save information in "cache" for the backward pass
        self.cache = A_prev.shape

        ### START CODE HERE ###
        A = A_prev.reshape(A_prev.shape[0], -1)
        ### END CODE HERE ###
        return A
    
    def backward(self, dA):
        """
        Implements the backward pass of the flatten layer

        Arguments:
        dA -- Input data, a 2-dimensional array

        Returns:
        dA_prev -- An array with its original shape (the output shape of its' previous layer).
        """
        ### START CODE HERE ###
        dA_prev = dA.reshape(self.cache)
        ### END CODE HERE ###
        return dA_prev