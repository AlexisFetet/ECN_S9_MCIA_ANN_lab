"""
This file contains a python script to construct a neural network to work on MNIST dataset
"""

import os

import numpy as np

TRAINING_IMAGES = os.path.abspath("../data/training/train-images-idx3-ubyte")
TRAINING_LABELS = os.path.abspath("../data/training/train-labels-idx1-ubyte")
TEST_IMAGES = os.path.abspath("../data/test/t10k-images-idx3-ubyte")
TEST_LABELS = os.path.abspath("../data/test/t10k-labels-idx1-ubyte")


class Layer():
    """
    represents a layer of the network
    """

    def __init__(self, connections_in_count, connections_out_count, func, d_func, alpha) -> None:
        v_gaussian = np.vectorize(
            lambda x: x * np.random.normal(0, 1/np.sqrt(connections_in_count)))
        self.weight = v_gaussian(
            np.ones((connections_out_count, connections_in_count)))
        self.bias = np.zeros((connections_out_count, 1))
        self.activation = func
        self.d_activation = d_func
        self.alpha = alpha
        self.last_z = np.zeros((connections_out_count, 1))
        self.last_a = np.zeros((connections_out_count, 1))
        self.last_delta = np.zeros((connections_out_count, 1))
        self.last_input = np.zeros((connections_in_count, 1))

    def forward(self, input_vector):
        """
        computes output of the layer
        """
        self.last_input = input_vector
        self.last_z = np.dot(self.weight, input_vector) + self.bias
        self.last_a = self.activation(self.last_z)
        return self.last_a

    def backward(self, delta, weight):
        """
        computes error based on previous layer error and weight
        """
        self.last_delta = np.dot(
            self.d_activation(self.last_z) * weight, delta)
        return self.last_delta

    def update(self):
        """
        update bias and weights
        """
        self.weight = self.weight - self.alpha * self.last_delta * self.last_input
        self.bias = self.bias - self.alpha * self.last_delta


def activation_function(input_vector) -> np.ndarray:
    """
    activation function
    """
    return 1/(1+np.exp(input_vector))


def d_activation_function(input_vector) -> np.ndarray:
    """
    derivative of the activation function
    """
    temp = activation_function(input_vector)
    return temp*(1-temp)


if __name__ == '__main__':

    