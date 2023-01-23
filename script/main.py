"""
This file contains a python script to construct a neural network to work on MNIST dataset
"""

import cProfile
import os
import random
from copy import deepcopy

import numpy as np
from mnist import MNIST

DATA_FOLDER = os.path.abspath("data")


class Layer():
    """
    represents a layer of the network
    """

    def __init__(self, connections_in_count, connections_out_count, func, d_func, alpha, m) -> None:
        v_gaussian = np.vectorize(
            lambda x: x * np.random.normal(0, 1/np.sqrt(connections_in_count)))
        self.weight = v_gaussian(
            np.ones((connections_out_count, connections_in_count)))
        self.bias = np.zeros((connections_out_count, 1))
        self.activation = func
        self.d_activation = d_func
        self.alpha = alpha
        self.m = m
        self.last_z = np.zeros((connections_out_count, 1))
        self.last_a = np.zeros((connections_out_count, 1))
        self.last_delta = np.zeros((connections_out_count, 1))
        self.last_input = np.zeros((connections_in_count, 1))

    def forward(self, input_vector):
        """
        computes output of the layer
        """
        self.last_input = input_vector
        self.last_z = np.dot(self.weight, input_vector) + \
            np.dot(self.bias, np.ones((1, self.m)))
        self.last_a = self.activation(self.last_z)
        return self.last_a

    def backward(self, delta, weight: np.ndarray):
        """
        computes error based on previous layer error and weight
        """
        self.last_delta = np.dot(weight.transpose(), delta) * \
            self.d_activation(self.last_z)
        return deepcopy(self.last_delta)

    def update(self):
        """
        update bias and weights
        """
        self.weight -= self.alpha/self.m * np.dot(
            self.last_delta, self.last_input.transpose())
        self.bias -= (self.alpha / self.m) * \
            np.dot(self.last_delta, np.ones((self.m, 1)))


def activation_function(input_vector) -> np.ndarray:
    """
    activation function
    """
    return 1/(1+np.exp(-1*input_vector))


def d_activation_function(input_vector) -> np.ndarray:
    """
    derivative of the activation function
    """
    temp = activation_function(input_vector)
    return temp*(1-temp)


class Network():
    """
    represents the network
    """

    def __init__(self, layers: list[Layer], m) -> None:
        self.layers: list[Layer] = layers
        self.m = m

    def compute(self, input_vector):
        """
        computes output of the ANN
        """
        temp = deepcopy(input_vector)
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def back_propagation(self, image_, expected: int):
        """
        update weight
        """
        result = self.compute(image_)
        expected_result = np.zeros(np.shape(result))
        for indx, expected_val in enumerate(expected):
            expected_result[expected_val, indx] = 1
        delta = result - expected_result
        for indx, layer in reversed(list(enumerate(self.layers))):
            if indx == len(self.layers)-1:
                delta = layer.backward(delta, np.eye(len(delta)))
            else:
                delta = layer.backward(delta, self.layers[indx + 1].weight)
        for layer in self.layers:
            layer.update()

    def result(self, image_):
        """
        compute the digit
        """
        output = self.compute(image_)
        output = [x[0] for x in output]
        return output.index(max(output))


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def img2arr(image_):
    """
    transforms image to vector
    """
    return np.c_[image_]  # mieux que np.array([[x] for x in image_])


def main():

    mndata = MNIST(DATA_FOLDER)
    training_images, training_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    M = 10

    layer1 = Layer(28*28, 30, activation_function,
                   d_activation_function, 0.05, M)
    layer2 = Layer(30, 10, activation_function, d_activation_function, 0.05, M)
    network = Network([layer1, layer2], M)

    # reference
    print("REFERENCE")
    correct = 0
    for (image, label) in list(zip(test_images, test_labels)):
        if network.result(img2arr(image)/255) == label:
            correct += 1
    print(f"correctness : {correct/len(test_images)*100}%")

    list_couples = list(
        zip([img2arr(image)/255 for image in training_images], training_labels))

    for train in range(30):
        random.shuffle(list_couples)
        for small_list_couples in grouped(list_couples, M):
            label = [elem[1] for elem in small_list_couples]
            images = [elem[0] for elem in small_list_couples]
            image_list = images[0]
            for image in images[1:]:
                image_list = np.c_[image_list, image]
            network.back_propagation(image_list, label)

        print(f"AFTER EPOCH {train + 1}")
        correct = 0
        for (image, label) in list(zip(test_images, test_labels)):
            if network.result(img2arr(image)/255) == label:
                correct += 1
        print(f"correctness : {correct/len(test_images)*100}%")


if __name__ == "__main__":
    cProfile.run('main()')
