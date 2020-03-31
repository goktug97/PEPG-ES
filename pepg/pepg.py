#!/usr/bin/env python3

import pickle
from typing import List
from functools import lru_cache

import numpy as np

from .optimizers import Adam, Optimizer

@lru_cache(maxsize=1)
def utility_function(population_size: int) -> List[float]:
    utility = np.clip(
        np.log(population_size/2 + 1) -
        np.log(np.arange(population_size, 0, -1)),
        0, None)
    utility /= np.sum(
        np.clip(np.log(population_size/2 + 1) -
                np.log(np.arange(1, population_size+1)),
                0, None))
    utility -= 1 / population_size
    return utility

@lru_cache(maxsize=1)
def center_function(population_size: int) -> List[float]:
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers

def compute_ranks(fitness_scores):
    fitness_scores = np.array(fitness_scores)
    ranks = np.empty(fitness_scores.size, dtype=int)
    ranks[fitness_scores.argsort()] = np.arange(fitness_scores.size)
    return ranks

def fitness_transformation(fitness_scores: List[float]) -> List[float]:
    ranks = compute_ranks(fitness_scores)
    population_size = len(fitness_scores)
    values = center_function(population_size)
    return values[ranks]


class PEPG():
    def __init__(self, population_size: int, theta_size: int,
                 mu_init: float, sigma_init: float, mu_lr: float,
                 sigma_lr: float, l2_coeff: float = 0.005,
                 optimizer = Adam, optimizer_kwargs = {}):
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
            
        assert issubclass(optimizer, Optimizer)
        self.optimizer = optimizer(mu_lr, **optimizer_kwargs)

    def get_parameters(self) -> List[List[float]]:
        self.epsilon = np.random.normal(
            0.0, self.sigma,
            (int(self.population_size/2), self.mu.size))
        self.theta = np.concatenate([self.mu + self.epsilon, self.mu - self.epsilon])
        return self.theta

    def update(self, rewards):
        idx = np.argmax(rewards)
        if rewards[idx] > self.best_fitness:
            self.best_mu = self.mu
            self.best_sigma = self.sigma
            self.best_theta = self.theta[idx]
            self.best_fitness = rewards[idx]

        rewards = fitness_transformation(rewards)
        baseline = np.mean(rewards)
        batch = int(self.population_size/2)
        s = ((self.epsilon ** 2) - (self.sigma ** 2)) / self.sigma
        rt = rewards[:batch] - rewards[batch:]
        rs = ((rewards[:batch] + rewards[batch:]) / 2 - baseline)
        mu_g = np.dot(self.epsilon.T, rt) / self.population_size
        self.mu += self.optimizer(-mu_g + self.l2_coeff * self.mu)
        self.sigma += self.sigma_lr * np.dot(s.T, rs) / self.population_size
        self.step += 1

    def save_checkpoint(self):
        import time
        import pathlib
        import os
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        folder = pathlib.Path(
            f"{os.path.join(os.path.dirname(module.__file__), 'checkpoints')}")
        folder.mkdir(parents=True, exist_ok=True)
        filename = f'{int(time.time())}.checkpoint'
        save_path = os.path.abspath(os.path.join(folder, filename))
        print(f'Saving checkpoint: {save_path}')
        with open(os.path.join(folder, filename), 'wb') as output:
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
