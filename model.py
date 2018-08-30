from typing import Tuple, Callable

import numpy as np
import gym
import gym.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal
#import torch.optim as optim

#1   = 2
#ADIM = 1
ADTYPE = np.float32
#3   = 8
SDTYPE = np.float32

device = torch.device("cuda")

class NormalizedActions(gym.ActionWrapper):
    def action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = l + (a + 1.0) * 0.5 * (h - l)
        a = np.clip(a, l, h)

        return a

    def reverse_action(self, a):
        l = self.action_space.low
        h = self.action_space.high

        a = 2 * (a -l)/(h - l) -1 
        a = np.clip(a, l, h)

        return a

class ScaledSigmoid(nn.Module):
    def __init__(self, bounds: Tuple[Tuple[float, ...], Tuple[float, ...]]):
        """
        A class for scaled sigmoid activation function.

        :param bounds:
            A tuple of two tuples of floats. The first element of the tuple (bounds[0]) is lower bounds. The second
            element of the tuple is upper bounds. As sigmoid: R -> [0, 1], scaling is done by
                (bounds[1][i] - bounds[0][i]) * x[i] + bounds[0][i]
            for each dimension of x = sigmoid(in).

        Usage:
            x = ScaledSigmoid(((-2, -2), (2, 2)))(in)
        """
        super(ScaledSigmoid, self).__init__()
        self.bounds = torch.tensor(bounds)
        self.scales = (self.bounds[1, :] - self.bounds[0, :])[None]

    def forward(self, x):
        x = torch.sigmoid(x)
        return self.scales * x + self.bounds[None, 0, :]


class RLNetwork(nn.Module):
    """
    An abstract class for neural networks in reinforcement learning (RL). In deep RL, many algorithms
    use DP algorithms. For example, DQN uses two neural networks: a main neural network and a target neural network.
    Parameters of a main neural network is periodically copied to a target neural network. This RLNetwork has a
    method called soft_update that implements this copying.
    """
    def __init__(self):
        super(RLNetwork, self).__init__()
        self.layers = []

    def forward(self, *x):
        return x

    def soft_update(self, target_nn: nn.Module, update_rate: float):
        """
        Update the parameters of the neural network by
            params1 = self.parameters()
            params2 = target_nn.parameters()

            for p1, p2 in zip(params1, params2):
                new_params = update_rate * p1.data + (1. - update_rate) * p2.data
                p1.data.copy_(new_params)

        :param target_nn:   DDPGActor used as explained above
        :param update_rate: update_rate used as explained above
        """

        params1 = self.parameters()
        params2 = target_nn.parameters()
        
        #bug? 
        for p1, p2 in zip(params1, params2):
            new_params = update_rate * p1.data + (1. - update_rate) * p2.data
            p1.data.copy_(new_params)

    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class SACActor(RLNetwork):
    def __init__(self, a_dim=1, n_neurons: Tuple[int, ...]=(3, 256, 256, 1), log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()
        self.n_layers = len(n_neurons)-2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        for i, (fan_in, fan_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(fan_in, fan_out)
            init.uniform_(layer.weight, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            init.uniform_(layer.bias, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            exec('self.fc{} = layer'.format(i+1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1])
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        self.output.bias.data.uniform_(-3e-3, 3e-3)

        self.log_std_output = nn.Linear(n_neurons[-2], n_neurons[-1])
        self.log_std_output.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_output.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = states
        for i in range(self.n_layers):
            x = eval('F.relu(self.fc{}(x))'.format(i+1))
        mu = self.output(x)
        log_std = self.log_std_output(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std

    def evaluate(self, state, eps=1e-6):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        z = normal.sample()
        action = torch.tanh(z)
        ###
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mu, log_std

    def get_action(self, s):
        s = torch.FloatTensor(s).unsqueeze(0).to(device)
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        z = normal.sample()
        a = torch.tanh(z)

        a = a.detach().cpu().numpy()
        return a[0]


class VCritic(RLNetwork):
    def __init__(self, n_neurons: Tuple[int, ...]=(3, 256, 256, 1)):
        super(VCritic, self).__init__()
        self.n_layers = len(n_neurons)-2
        
        for i, (fan_in, fan_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(fan_in, fan_out)
            init.uniform_(layer.weight, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            init.uniform_(layer.bias, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            exec('self.fc{} = layer'.format(i+1))
        
        layer = nn.Linear(n_neurons[-2], n_neurons[-1])
        init.uniform_(layer.weight, -3e-3, 3e-3)
        init.uniform_(layer.bias, -3e-3, 3e-3)
        self.output = layer

    def forward(self, states: torch.Tensor):
        x = states
        for i in range(self.n_layers):
            x = eval('F.relu(self.fc{}(x))'.format(i+1))
        
        return self.output(x)


class SoftQCritic(RLNetwork):
    def __init__(self, a_dim=1, n_neurons: Tuple[int, ...]=(3, 256, 256, 1)):
        super(SoftQCritic, self).__init__()
        self.n_layers = len(n_neurons)-2

        for i, (fan_in, fan_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            fan_in = fan_in + a_dim if i == 0 else fan_in
            layer = nn.Linear(fan_in, fan_out)
            init.uniform_(layer.weight, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            init.uniform_(layer.bias, -1./np.sqrt(fan_in), 1./np.sqrt(fan_in))
            exec('self.fc{} = layer'.format(i+1))

        layer = nn.Linear(n_neurons[-2], n_neurons[-1])
        init.uniform_(layer.weight, -3e-3, 3e-3)
        init.uniform_(layer.bias, -3e-3, 3e-3)
        self.output = layer

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([states, actions], dim=1)
        for i in range(self.n_layers):
            x = eval('F.relu(self.fc{}(x))'.format(i+1))
        
        return self.output(x)
    


