from abc import ABC, abstractmethod

import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def __init__(self, alpha):
        pass

    @abstractmethod
    def __call__(self, gradients):
        pass

class Adam(Optimizer):
    def __init__(self, alpha: float, beta_1: float = 0.9, beta_2:
                 float = 0.999, epsilon: float = 1e-08):

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None

        self.t = 0

    def __call__(self, gradients):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(gradients)
            self.v = np.zeros_like(gradients)
        alpha = self.alpha * (np.sqrt(1 - np.power(self.beta_2, self.t)) /
                              (1 - np.power(self.beta_1, self.t)))
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradients
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradients)
        return -alpha * self.m / (np.sqrt(self.v) + self.epsilon)

