import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, lr, in_dim, out_dim, fc1=512, fc2=128):

        super(DQN, self).__init__()
        self.in_dim = in_dim[0]
        self.out_dim = out_dim
        self.fc_input_dim = 2048
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, self.out_dim)
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        actions = self.fc(x)
        return actions

class ReplayBuffer(object):
    'Experience replay implementation'
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.pointer = 0
        self.state_memory = np.zeros((self.mem_size, 4, 80, 80), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 4, 80, 80), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def push(self, state, action, reward, new_state, done):
        index = self.pointer % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done

        self.pointer += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.pointer, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states,  actions, rewards, new_states, dones

class Agent:
    def __init__(self, gamma, eps, lr, input_dims,
                 batch_size, n_actions, max_mem_size=100000, eps_end=0.02, eps_dec=5e-4):
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.input_dims = input_dims

        self.Q_eval = DQN(lr, self.input_dims,
                            n_actions, 256, 256)
        self.Q_target = DQN(lr, self.input_dims, n_actions, 256, 256)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_eval.to(self.Q_eval.device)
        self.Q_target.to(self.Q_target.device)
        self.step_count = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = torch.tensor([observation], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        if self.step_count%1000 == 0: # change target manually like this
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            print('Step {}: Target Q-Network replaced!'.format(self.step_count))
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0    # value of terminal states identically zero

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min
        self.step_count += 1

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