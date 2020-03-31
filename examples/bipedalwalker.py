import time

import numpy as np
import gym
gym.logger.set_level(40)

from pepg import PEPG, NeuralNetwork

env = gym.make('BipedalWalker-v3')

network = NeuralNetwork(env.observation_space.shape[0],
                        env.action_space.shape[0],
                        [64, 64], hidden_activation = np.tanh)

es = PEPG(population_size = 256, theta_size = network.number_of_parameters,
          mu_init = 0.0, sigma_init = 2.0,
          mu_lr = 0.3, sigma_lr = 0.2)

try:
    while True:
        solutions = es.get_parameters()
        rewards = []
        prev_time = time.time()
        print(f'Step: {es.step}')
        for solution in solutions:
            network.weights = solution
            done = False
            observation = env.reset()
            total_reward = 0.0
            while not done:
                output = network(observation)
                observation, reward, done, info = env.step(output)
                total_reward += reward
            rewards.append(total_reward)
        es.update(rewards)
        print(f'Step Took: {time.time() - prev_time} seconds')
        print(f'Max Reward Session: {es.best_fitness}')
        print(f'Max Reward Step: {np.max(rewards)}')
except KeyboardInterrupt:
    env = gym.wrappers.Monitor(env, '.', force = True)
    observation = env.reset()
    done = False
    total_reward = 0
    network.weights = es.best_theta
    while not done:
        output  = network(observation)
        observation, reward, done, info = env.step(output)
        total_reward += reward
        env.render()
    print(f'Reward: {total_reward}')
finally:
    es.save_checkpoint()
    env.close()
    
