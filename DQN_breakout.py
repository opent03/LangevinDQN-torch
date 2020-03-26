'''
A basic DQN to play breakout. Uses epsilon-greedy actions.
'''
from collections import namedtuple
import random
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import torch.nn as nn
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, lr):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_input_dim = 3072
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_dim, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.out_dim)
        )

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float).to(device)
        x = x.view(-1, 1, 105, 80)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        qvals = self.fc(x)
        return qvals


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    ''' From Pytorch documentation '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LangevinOptimizer(Optimizer):
    ''' Implements Langevin SGD updates to params '''
    def __init__(self, params, lr=1e-3):
        self.lr = lr
        super(LangevinOptimizer, self).__init__(params)

    @torch.no_grad()
    def step(self,  closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                d_p = p.grad

class Agent:
    ''' Specific agents will inherit this class '''
    def __init__(self, gamma, epsilon, lr, in_dim, out_dim, batch_size=32,
                 mem_capacity=1000000, eps_min=0.01, eps_dec=0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.action_space = [i for i in range(self.out_dim)]
        self.memory = ReplayMemory(mem_capacity)
        self.Q_eval = DQN(in_dim=in_dim, out_dim=out_dim, lr=lr)

        # hard code target here
        self.Q_target = DQN(in_dim=in_dim, out_dim=out_dim, lr=lr)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_eval.to(device)
        self.Q_target.to(device)
        self.steps = 0

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice([0,1,2,3])
        else:
            actions = self.Q_eval.forward(observation)
            action = torch.argmax(actions[1]).item()
        return action

    def learn(self):
        if self.memory.position > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            batch = self.memory.sample(batch_size=self.batch_size)
            batch = Transition(*zip(*batch))    # ??????????????????????

            state_batch = np.asarray(batch.state)
            action_batch = np.asarray(batch.action)
            reward_batch = np.asarray(batch.reward)
            next_state_batch = np.asarray(batch.next_state)

            #print(" state batch shape ", state_batch.shape)
            #print(" next state batch shape ", next_state_batch.shape)
            if self.steps % 1000:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
            q_eval = self.Q_eval.forward(state_batch).to(device)
            q_target = self.Q_target.forward(state_batch).to(device)
            q_next = self.Q_eval.forward(next_state_batch).to(device)

            max_actions = torch.argmax(q_next, dim=1).to(device)
            rewards = torch.tensor(reward_batch, dtype=torch.float).to(device)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, max_actions] = rewards + self.gamma*torch.max(q_next, dim=1)[0]

            # custom epsilon
            if self.steps > 5000:
                if self.epsilon - 1e-5 > self.eps_min:
                    self.epsilon -= 1e-5
                else:
                    self.epsilon=self.eps_min
            loss = self.Q_eval.loss(q_target, q_eval).to(device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.steps += 1