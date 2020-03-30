#!/usr/bin/env python3

from pepg import PEPG, NeuralNetwork, Adam, sigmoid
import numpy as np


network = NeuralNetwork(input_size = 2, output_size = 1, hidden_sizes = [2],
                        hidden_activation = sigmoid,
                        output_activation = sigmoid)

# Adam Optimizer is the default optimizer, it is written for the example
optimizer_kwargs = {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08} # Adam Parameters

es = PEPG(population_size = 100, theta_size = network.number_of_parameters,
          mu_init = 0, sigma_init = 2.0,
          mu_lr = 0.3, sigma_lr = 0.2, optimizer = Adam,
          optimizer_kwargs = optimizer_kwargs)

truth_table = [[0, 1],[1, 0]]
solution_found = False

while True:
    print(f'Step: {es.step}')
    solutions = es.get_parameters()
    rewards = []
    for solution in solutions:
        network.weights = solution
        error = 0
        for input_1 in range(len(truth_table)):
            for input_2 in range(len(truth_table[0])):
                output = int(round(network([input_1, input_2])[0]))
                error += abs(truth_table[input_1][input_2] - output)
        reward = (4 - error) ** 2
        rewards.append(reward)
    es.update(rewards)
    if es.best_fitness == 16:
        print('Solution Found')
        print(f'Parameters: {es.best_theta}')
        break
