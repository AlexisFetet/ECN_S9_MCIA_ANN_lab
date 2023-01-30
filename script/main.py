"""
This file contains a python script to construct a neural network to work on MNIST dataset
Alexis Fetet
Sanaa Sghir
"""

import cProfile
import os
import pstats
import random
from copy import deepcopy

import numpy as np
from mnist import MNIST


class Layer():
    """
    represents a layer of the network
    """

    def __init__(self, connections_in_count, connections_out_count, func, d_func, alpha, m) -> None:
        # create an element-wise function to initialize the weigths
        v_gaussian = np.vectorize(
            lambda x: x * np.random.normal(0, 1/np.sqrt(connections_in_count)))
        # initialize the weigth with gaussian
        self.weight = v_gaussian(
            np.ones((connections_out_count, connections_in_count)))
        self.bias = np.zeros((connections_out_count, 1))  # initialize bias
        self.activation = func  # initialize the activation function of the layer
        # initialize  derivative of the activation function of the layer
        self.d_activation = d_func
        self.alpha = alpha  # initialize  learning rate
        self.m = m  # initialize the value m, number of image per batch
        # initialize the various variables for backpropagation and weight and bias updates
        self.last_z = np.zeros((connections_out_count, 1))
        self.last_delta = np.zeros((connections_out_count, 1))
        self.last_input = np.zeros((connections_in_count, 1))
        # initialise special 1 vector
        self.one = np.ones((1, self.m))
        self.one_transpose = np.ones((self.m, 1))

    def forward(self, input_vector):
        """
        computes output of the layer
        """
        self.last_input = input_vector  # save input (a l-1) for weight update
        self.last_z = np.dot(self.weight, input_vector) + \
            np.dot(self.bias, self.one)  # use given formula
        return self.activation(self.last_z)  # compute activation

    def backward(self, delta, weight: np.ndarray):
        """
        computes error based on previous layer error and weight
        """
        self.last_delta = np.dot(weight.transpose(), delta) * \
            self.d_activation(self.last_z)  # use given formula
        # the delta value is passed from layer to layer, we need a copy of it
        return deepcopy(self.last_delta)

    def update(self):
        """
        update bias and weights
        """
        self.weight -= self.alpha/self.m * np.dot(
            self.last_delta, self.last_input.transpose())  # use given weight update formula
        self.bias -= (self.alpha / self.m) * \
            np.dot(self.last_delta, self.one_transpose
                   )  # use given bias update formula


def activation_function(input_vector) -> np.ndarray:
    """
    activation function
    """
    return 1/(1+np.exp(-1*input_vector))  # np.exp is element-wise


def d_activation_function(input_vector) -> np.ndarray:
    """
    derivative of the activation function
    """
    temp = activation_function(
        input_vector)  # calcule 1 fois la fonction d'activation
    return temp*(1-temp)


class Network():
    """
    represents the network
    """

    def __init__(self, layers: list[Layer], m) -> None:
        self.layers: list[Layer] = layers
        self.m = m
        # initialise eye matrix for weight values used for backpropagation in output layer
        self.eye = np.eye(len(layers[-1].weight))

    def compute(self, input_vector):
        """
        computes output of the ANN
        """
        temp = input_vector  # initialize the vector to be passed from layer to layer
        for layer in self.layers:
            # progress throught the network's layers
            temp = layer.forward(temp)
        return temp

    def learn(self, image_, expected: int):
        """
        update weight
        """
        result = self.compute(
            image_)  # compute the reult using current networks
        # initialize expected result
        expected_result = np.zeros(np.shape(result))
        for indx, expected_val in enumerate(expected):
            expected_result[expected_val, indx] = 1
        # compute delta of output layer
        delta = result - expected_result
        # propagate error to first layer, layers will remember their error
        for indx, layer in reversed(list(enumerate(self.layers))):
            if indx == len(self.layers)-1:
                delta = layer.backward(delta, self.eye)  # this is output layer
            else:
                # give necessary info for layer indx to compute it's error
                delta = layer.backward(delta, self.layers[indx + 1].weight)
        # update all the layer's weight and bias using newly calculated errors
        for layer in self.layers:
            layer.update()

    def result(self, image_):
        """
        compute the digit
        """
        output = self.compute(image_)
        output = [x[0] for x in output]
        # return index of the most probable output, in our case that's also the digit the network thinks is on the picture
        return output.index(max(output))


def grouped(iterable, n):
    """
    group iterable element n by n
    """
    # here, there is a single instance of iter created, with n references to it
    # when zipping, zip effectively calls the __next__ method on each of the n
    # references, indeed grouping elements n by n
    return zip(*[iter(iterable)]*n)


def img2arr(image_):
    """
    transforms image to vector
    """
    return np.c_[image_]  # transforms list in column vector


def main():
    """
    main program
    """
    mndata = MNIST(os.path.abspath("data"))  # load images located in ./data
    # mndata = MNIST(os.path.abspath(""))  # load images located in current directory
    training_images, training_labels = mndata.load_training()  # load training data
    test_images, test_labels = mndata.load_testing()  # load test data
    M = 10  # set mini batch size to 10

    layer1 = Layer(28*28, 30, activation_function,
                   d_activation_function, 0.05, M)  # create first layer
    layer2 = Layer(30, 10, activation_function,
                   d_activation_function, 0.05, M)  # create second layer
    network = Network([layer1, layer2], M)  # create network from both layers

    # convert test images to vector
    test_images = [img2arr(image)/255 for image in test_images]
    # convert training images to vectors
    # list_couples is a list of pairs (image, label) so we can safely shuffle
    list_couples = list(
        zip([img2arr(image)/255 for image in training_images], training_labels))

    # test network with no training
    print("REFERENCE")
    correct = 0
    for (image, label) in list(zip(test_images, test_labels)):
        # if the network gets it correctly, keep track of it
        if network.result(image) == label:
            correct += 1
    print(f"correctness : {correct/len(test_images)*100}%")

    for train in range(30):
        random.shuffle(list_couples)  # shuffle data
        for small_list_couples in grouped(list_couples, M):
            # recreate lists of images and labels
            label = [elem[1] for elem in small_list_couples]
            images = [elem[0] for elem in small_list_couples]
            # concatenate all images in a big matrix
            image_list = images[0]
            for image in images[1:]:
                image_list = np.c_[image_list, image]
            # apply learning
            network.learn(image_list, label)

        # test network
        print(f"AFTER EPOCH {train + 1}")
        correct = 0
        for (image, label) in list(zip(test_images, test_labels)):
            if network.result(image) == label:
                correct += 1
        print(f"correctness : {correct/len(test_images)*100}%")


if __name__ == "__main__":
    # using the profiler let's us see what are the most costly function calls of the program
    # allowed us to reduce the compute time from over 5min down to under 100s (with profiler not enabled)
    profiler = cProfile.Profile(builtins=False)
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
