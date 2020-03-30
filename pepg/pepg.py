#!/usr/bin/env python3

import pickle
from typing import List
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def utility_function(population_size: int) -> List[float]:
    utility = np.clip(
        np.log(population_size/2 + 1) -
        np.log(np.arange(1, population_size+1)),
        0, None)
    utility /= np.sum(
        np.clip(np.log(population_size/2 + 1) -
                np.log(np.arange(1, population_size+1)),
                0, None))
    utility -= 1 / population_size
    return utility

def fitness_transformation(fitness_scores: List[float]) -> List[float]:
    ranks = np.argsort(fitness_scores)[::-1]
    population_size = len(fitness_scores)
    utility = utility_function(population_size)
    return utility[ranks]


class Adam():
    def __init__(self, parameters_size: int, alpha: float,
                 beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-08):

        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 0
        self.m = np.zeros(parameters_size)
        self.v = np.zeros(parameters_size)

    def __call__(self, gradients):
        self.t += 1
        alpha = self.alpha * (np.sqrt(1 - np.power(self.beta_2, self.t)) /
                              (1 - np.power(self.beta_1, self.t)))
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradients
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradients)
        return -alpha * self.m / (np.sqrt(self.v) + self.epsilon)


class PEPG():
    def __init__(self, population_size: int, theta_size: int,
                 mu_init: float, sigma_init: float, mu_lr: float,
                 sigma_lr: float, l2_coeff: float = 0.005,
                 optimizer = Adam):
        self.population_size: int = population_size
        self.theta_size: int = theta_size
        self.mu_init: float = mu_init
        self.sigma_init: float = sigma_init
        self.mu_lr: float = mu_lr
        self.sigma_lr: float = sigma_lr
        self.l2_coeff: float = l2_coeff

        self.step = 0

        self.mu = mu_init * np.ones(theta_size)
        self.sigma = sigma_init * np.ones(theta_size)

        self.best_fitness = -float('inf')
        self.best_mu = None
        self.best_sigma = None
        self.best_theta = None

        self.optimizer = optimizer(theta_size, alpha=mu_lr)

    def get_parameters(self) -> List[List[float]]:
        self.epsilon = np.random.normal(
            0.0, self.sigma,
            (int(self.population_size/2), self.mu.size))
        self.theta = np.concatenate([self.mu + self.epsilon, self.mu - self.epsilon])
        return self.theta

    def update(self, results):
        idx = np.argmax(results)
        if results[idx] > self.best_fitness:
            self.best_mu = self.mu
            self.best_sigma = self.sigma
            self.best_theta = self.theta[idx]
            self.best_fitness = results[idx]

        results = fitness_transformation(results)
        baseline = np.mean(results)
        batch = int(self.population_size/2)
        s = ((self.epsilon ** 2) - (self.sigma ** 2)) / self.sigma
        rt = results[:batch] - results[batch:]
        rs = ((results[:batch] + results[batch:]) / 2 - baseline)
        mu_g = np.dot(self.epsilon.T, rt) / self.population_size
        self.mu += self.optimizer(-mu_g + self.l2_coeff * self.mu)
        self.sigma += self.sigma_lr * np.dot(s.T, rs) / self.population_size
        self.step += 1

    def save_checkpoint(self):
        import time
        filename = f'{int(time.time())}.checkpoint'
        with open(filename, 'wb') as output:
            pickle.dump(self.__dict__, output, -1)

    @classmethod
    def load_checkpoint(cls, filename):
        with open(filename, 'rb') as checkpoint:
            cls_dict = pickle.load(checkpoint)
        es = cls.__new__(cls)
        es.__dict__.update(cls_dict)
        return es

    def save_best(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump((self.best_theta, self.best_mu, self.best_sigma), output, -1)

    @classmethod
    def load_best(cls, filename):
        with open(filename, 'rb') as best:
            theta, mu, sigma = pickle.load(best)
        return theta, mu, sigma
