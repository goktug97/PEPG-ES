import pickle
from typing import Callable, List

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


class NeuralNetwork(object):
    def __init__(self, input_size:int, output_size: int, hidden_sizes: List[int] = [],
                 hidden_activation: Callable[[float], float] = lambda x: x,
                 output_activation: Callable[[float], float] = lambda x: x,
                 bias: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.bias = bias

        sizes = np.concatenate([[input_size], hidden_sizes, [output_size]])
        number_of_parameters = 0
        for i in range(len(sizes)-1):
            number_of_parameters += sizes[i] * sizes[i+1]
        if bias:
            number_of_parameters += np.sum(sizes[1:])
        self.number_of_parameters: int = int(number_of_parameters)
        self.sizes = sizes.astype('int')

        self._weights = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)

    def __call__(self, value):
        assert self._weights is not None

        weights = self.weights
        output = value
        if self.bias:
            output = np.append(output, 1)
        index = 0
        for i in range(len(self.sizes)-2):
            w = weights[index:index+self.sizes[i]*self.sizes[i+1]+self.sizes[i+1]]
            w = w.reshape(self.sizes[i]+1, self.sizes[i+1])
            index += self.sizes[i]*self.sizes[i+1]+self.sizes[i+1]
            output = self.hidden_activation(output @ w)
            if self.bias:
                output = np.append(output, 1)

        w = weights[index:]
        w = w.reshape(self.sizes[len(self.sizes)-2]+(1 if self.bias else 0),
                      self.sizes[len(self.sizes)-1])
        output = self.output_activation(output @ w)
        return output

    def save_network(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load_network(cls, filename):
        with open(filename, 'rb') as f:
            cls_dict = pickle.load(f)
        network = cls.__new__(cls)
        network.__dict__.update(cls_dict)
        return network
