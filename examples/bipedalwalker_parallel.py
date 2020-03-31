import os
import subprocess
import sys
import time

from mpi4py import MPI
import gym
import numpy as np
gym.logger.set_level(40)

from pepg import PEPG, NeuralNetwork, tanh

# Working prototype for parallel ES.

def fork(n_proc):
    if os.getenv('MPI_PARENT') is None:
        env = os.environ.copy()
        env['MPI_PARENT'] = '1'
        subprocess.call(['mpirun', '-use-hwthread-cpus', '-np',
            str(n_proc), sys.executable, '-u', __file__], env=env)
        return True
    return False


class ParallelES():
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_workers = self.comm.Get_size() 

        self.network = NeuralNetwork(self.env.observation_space.shape[0],
                                     self.env.action_space.shape[0],
                                     [64, 64], hidden_activation = tanh,
                                     output_activation = tanh)

        self.job_per_worker = 32
        if self.rank == 0:
            self.es = PEPG(population_size = self.n_workers * self.job_per_worker,
                           theta_size = self.network.number_of_parameters,
                           mu_init = 0.0, sigma_init = 2.0,
                           mu_lr = 0.3, sigma_lr = 0.2)

    def eval_solutions(self, solutions):
        rewards = []
        for solution in solutions:
            self.network.weights = solution
            total_reward = 0.0
            observation = self.env.reset()
            done = False
            while not done:
                output = self.network(observation)
                observation, reward, done, info = self.env.step(output)
                total_reward += reward
            rewards.append(total_reward)
        return rewards

    def master(self):
        while True:
            prev_time = time.time()
            print(f'Step: {self.es.step}')
            solutions = self.es.get_parameters()
            split_solutions = np.array_split(solutions, self.n_workers)
            for i in range(1, self.n_workers):
                self.comm.Send(split_solutions[i], dest=i)
            rewards = np.empty(self.n_workers * self.job_per_worker)
            rewards[:self.job_per_worker] = self.eval_solutions(split_solutions[0])
            for i in range(1, self.n_workers):
                self.comm.Recv(rewards[
                    i*self.job_per_worker:i*self.job_per_worker+self.job_per_worker],
                               source=i)
            self.es.update(list(rewards))
            print(f'Step Took: {time.time() - prev_time} seconds')
            print(f'Max Reward Session: {self.es.best_fitness}')
            print(f'Max Reward Step: {np.max(rewards)}')
            if not (self.es.step % 100):
                self.network.weights = self.es.best_theta
                self.network.save_network('best_network.pickle')
                self.es.save_checkpoint()

    def slave(self):
        while True:
            solutions = np.empty((self.job_per_worker,
                                  self.network.number_of_parameters))
            self.comm.Recv(solutions, source=0)
            rewards = self.eval_solutions(solutions)
            self.comm.Send(np.array(rewards), dest=0)

    def main(self):
        if fork(n_proc = 4): sys.exit(0)
        self.master() if self.rank == 0 else self.slave()
    

if __name__ == '__main__':
    es = ParallelES()
    es.main()
