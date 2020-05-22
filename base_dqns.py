'''
A basic DQN with experience replay and fixed TD targets.
Tailored to the bsuite DeepSea environment.
'''
from collections import namedtuple
import random
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    ''' Basic Implementation of DQN '''
    def __init__(self, in_dim, out_dim, lr, langevin=False, name='eval'):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 2)
        self.name = name
        self.optimizer = optim.Adam(self.parameters(), lr=lr) if not langevin else LangevinOptimizer((self.parameters()))
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        qvals = self.fc3(x)
        return qvals


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))

class ReplayMemory(object):
    ''' From Pytorch documentation '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.counter = 0        # gives raw count of how many things stored

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class EpsilonAgent:
    def __init__(self, gamma, eps, lr, input_dims, output_dims,
                 batch_size, n_actions, max_mem_size=100000, eps_end=0.02, eps_dec=5e-4, target_update=200):
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.target_update = target_update

        # DQNs
        self.Q_eval = DQN(in_dim=self.input_dims, out_dim=self.output_dims, lr=self.lr, langevin=False)
        self.Q_target = DQN(in_dim=self.input_dims, out_dim=self.output_dims, lr=self.lr, langevin=False, name='target')
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.Q_target.to(self.Q_target.device)
        self.Q_eval.to(self.Q_eval.device)

        self.replay_memory = ReplayMemory(self.max_mem_size)

    def store_transition(self, state, action, reward, state_, done):
        assert state.shape == (100,), state_.shape == (100,)
        self.replay_memory.push(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = torch.tensor(observation, dtype=torch.float32).to(self.Q_eval.device)
            q_vals = self.Q_eval.forward(state)
            action = torch.argmax(q_vals).item()
        else:
            action = np.random.choice(self.n_actions)

        return action

    def learn(self):
        if self.replay_memory.counter < self.batch_size:
            return
        if self.replay_memory.counter % self.target_update == 0:  # change target manually like this
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            print('Step {}: Target Q-Network replaced!'.format(self.replay_memory.counter))
        self.Q_eval.optimizer.zero_grad()

        # we retrieve random batch and put them into the correct type
        batch = self.replay_memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = [],[],[],[],[]
        for i in range(len(batch)):
            assert batch[i].state.shape == (100,), batch[i].next_state.shape == (100,)
            state_batch.append(batch[i].state)
            action_batch.append(batch[i].action)
            reward_batch.append(batch[i].reward)
            new_state_batch.append(batch[i].next_state)
            terminal_batch.append(batch[i].terminal)

        state_batch = torch.tensor(state_batch).to(self.Q_eval.device)
        action_batch = torch.tensor(action_batch).to(self.Q_eval.device)
        reward_batch = torch.tensor(reward_batch).to(self.Q_eval.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.Q_eval.device)
        terminal_batch = torch.tensor(terminal_batch).to((self.Q_eval.device))
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end else self.eps_end

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
        '''
        for group in self.param_groups:
            for p in group['params']:
                d_p = p.grad12
        '''