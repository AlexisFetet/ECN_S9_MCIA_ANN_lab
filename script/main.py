import os
import numpy as np

TRAINING_IMAGES = os.path.abspath("../data/training/train-images-idx3-ubyte")
TRAINING_LABELS = os.path.abspath("../data/training/train-labels-idx1-ubyte")
TEST_IMAGES = os.path.abspath("../data/test/t10k-images-idx3-ubyte")
TEST_LABELS = os.path.abspath("../data/test/t10k-labels-idx1-ubyte")

class Layer():

    def __init__(self, connections_in_count, connections_out_count, func, d_func, alpha) -> None:
        self.weight = np.ones((connections_out_count, connections_in_count)) * np.random.normal(0, 1/np.sqrt(connections_in_count))
        self.bias = np.zeros((connections_out_count, 1))
        self.activation = func
        self.d_activation = d_func
        self.alpha = alpha
        self.last_z = np.zeros((connections_out_count, 1))
        self.last_a = np.zeros((connections_out_count, 1))
        self.last_delta = np.zeros((connections_out_count, 1))
        self.last_input = np.zeros((connections_in_count, 1))

    def forward(self, input_vector):
        self.last_input = input_vector
        self.last_z = np.dot(self.weight, input_vector) + self.bias
        self.last_a = self.activation(self.last_z)
        return self.last_a

    def backward(self, delta, weight):
        self.last_delta = np.dot(self.d_activation(self.last_z) * weight, delta)
        return self.last_delta
    
    def update(self):
        self.weight = self.weight - self.alpha * self.last_delta * self.last_input
        self.bias = self.bias - self.alpha * self.last_delta


def activation_function(x) -> np.ndarray:
    return 1/(1+np.exp(x))

def d_activation_function(x) -> np.ndarray:
    temp = activation_function(x)
    return temp*(1-temp)




if __name__ == '__main__':
    

    ALPHA = 1.1  # pas d'apprentissage

    w1 = np.array([[0.9, 1.1],
                [1, 1]])

    w2 = np.array([[1, 1],
                [1, 1]])

    b1 = np.array([[-1],
                [-2]])

    b2 = np.array([[-1],
                [-1]])

    a0 = np.array([[1],
                [1]])

    y = np.array([[0],
                [1]])

    a2 = np.array([[0],
                [0]])

    while(abs(sum(a2-y)) > 1e-5):
        # calcul de la sortie
        z1 = np.dot(w1, a0) + b1
        a1 = activation_function(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = activation_function(z2)
        # back propagation
        delta2 = d_activation_function(z2) * (a2-y)
        delta1 = np.dot(d_activation_function(z1) * w2, delta2)

        # mise à jour des poids
        w1 = w1 - ALPHA * delta1 * a0.transpose()
        w2 = w2 - ALPHA * delta2 * a1.transpose()

        # mise à jour des biais
        b1 = b1 - ALPHA * delta1
        b2 = b2 - ALPHA * delta2

    print(w1)
    print(b1)

    print(w2)
    print(b2)
