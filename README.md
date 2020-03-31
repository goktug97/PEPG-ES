Parameter-exploring Policy Gradients
=======================================================

Python Implementation of Parameter-exploring Policy Gradients <a href="#sehnke2010">[3]</a> Evolution Strategy 

## Requirements
* Python >= 3.6
* Numpy

### Optional
* gym
* mpi4py

## Install

- From PyPI

``` bash
pip3 install pepg-es
```

- From Source

``` bash
git clone https://github.com/goktug97/PEPG-ES
cd PEPG-ES
python3 setup.py install --user
```

## About Implementation

I implemented several things differently from the original paper;

- Applied rank transformation <a href="#wierstra14a">[1]</a> to the fitness scores.
- Used Adam <a href="#kingma2014adam">[2]</a> optimizer to update the mean.
- Weight decay is applied to the mean, similar to <a href="#salimans2017evolution">[4]</a>.

## Usage

Refer to [PEPG-ES/examples](https://github.com/goktug97/PEPG-ES/blob/master/examples)
folder for more complete examples.

### XOR Example

* Find Neural Network parameters for XOR Gate. 
* Black-box optimization algorithms like PEPG are competitive in the
  area of reinforcement learning because they don't require
  backpropagation to calculate the gradients.  In supervised learning
  using backpropagation is faster and more reliable. Thus, using backpropagation
  to solve the XOR problem would be faster. I demonstrated library by solving XOR
  because it was easy and understandable.

``` python
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
```

* Output:

``` bash
Step: 233
Step: 234
Step: 235
Step: 236
Step: 237
Solution Found
Parameters: [ 1.25863047 -0.73151503 -2.53377723  1.01802355  3.02723507  1.23112726
 -2.00288859 -3.66789242  4.56593794]
```

## Documentation

### PEPG Class

``` python

es = PEPG(self, population_size, theta_size,
          mu_init, sigma_init, mu_lr,
          sigma_lr, l2_coeff = 0.005,
          optimizer = Adam, optimizer_kwargs = {})

```

* **Parameters:**
    - **population_size:** int: Population size of the evolution strategy.
    - **theta_size** int: Number of parameters that will be optimized.
    - **mu_init** float: Initial mean.
    - **sigma_init** float: Initial sigma.
    - **mu_lr** float: Learning rate for the mean.
    - **sigma_lr** float: Learning rate for the sigma.
    - **l2_coeff** float: Weight decay coefficient.
    - **optimizer** Optimizer: Optimizer to use
    - **optimizer_kwargs** Dict[str, Any]: Parameters for optimizer except learning rate.

___

``` python
solutions = self.get_parameters(self)
```

- Creates symmetric samples around the mean and returns a numpy array with the size of
**[population_size, theta_size]**

___

``` python
self.update(self, rewards)
```

* **Parameters:**
    - **rewards:** List[float]: Rewards for the given solutions.
    
- Update the mean and the sigma.

___

``` python
self.save_checkpoint(self)
```

- Creates a checkpoint and save it into created time.time().checkpoint file.

___

``` python
es = PEPG.load_checkpoint(cls, filename)
```

- Creates a new PEPG class and loads the checkpoint.
___

``` python
self.save_best(self, filename)
```

- Saves the best theta and the mu and the sigma that used to create the best theta.

___

``` python
theta, mu, sigma = PEPG.load_best(cls, filename)
```

- Load the theta, the mu, and the sigma arrays from the given file.

### NeuralNetwork Class

``` python

NeuralNetwork(self, input_size, output_size, hidden_sizes = [],
              hidden_activation = lambda x: x,
              output_activation = lambda x: x,
              bias = True):

```

* **Parameters:**
    - **input_size:** int: Input size of network.
    - **output_size:** int: Output size of the network.
    - **hidden_sizes:** List[int]: Sizes for the hidden layers.
    - **hidden_activation:** Callable[[float], float]: Activation function used in hidden layers.
    - **output_activation:** Callable[[float], float]: Activation function used at the output.
    - **bias:** bool: Add bias node.
___

``` python
self.save_network(self, filename)
```

- Save the network to a file.

___

``` python
network = NeuralNetwork.load_network(cls, filename)
```

- Creates a new NeuralNetwork class and loads the given network file.
    
### Custom Optimizer Example

``` python
from pepg import PEPG, Optimizer, NeuralNetwork

class CustomOptimizer(Optimizer):
    def __init__(self, alpha, parameter, another_parameter):
        self.alpha = alpha
        self.parameter = parameter
        self.another_parameter = another_parameter

    def __call__(self, gradients):
        gradients = (gradients + self.parameter) * self.another_parameter
        return -self.alpha * gradients

network = NeuralNetwork(input_size = 2, output_size = 1)

optimizer_kwargs = {'parameter': 0.3, 'another_parameter': 0.2}
es = PEPG(population_size = 100, theta_size = network.number_of_parameters,
          mu_init = 0.0, sigma_init = 2.0,
          mu_lr = 0.3, sigma_lr = 0.2, optimizer = CustomOptimizer,
          optimizer_kwargs = optimizer_kwargs)
```

## References
1. <a id="wierstra14a"></a>Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters and Jurgen Schmidhuber. Natural Evolution Strategies. 2014
2. <a id="kingma2014adam"></a>Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. 2014
3. <a id="sehnke2010"></a>F. Sehnke, C. Osendorfer, T. Ruckstiess, A. Graves, J. Peters and J. Schmidhuber. Parameter-exploring policy gradients. 2010
4. <a id="salimans2017evolution"></a>Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor and Ilya Sutskever. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. 2017
