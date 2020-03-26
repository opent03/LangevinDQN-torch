import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, lr, fc1=512, fc2=128):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_input_dim = 3072
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
    '''
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float).to(device)
        x = x.view(-1, 1, 105, 80)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        qvals = self.fc(x)
        return qvals
    '''
    def forward(self, state):
        x = self.conv(state)
        action = self.fc(x)




class ReplayBuffer(object):
    'Experience replay implementation'
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.pointer = 0
        self.state_memory = np.zeros(self.mem_size, *input_shape, dtype=np.float32)
        self.new_state_memory = np.zeros(self.mem_size, *input_shape, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

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
    def __init__(self, lr, gamma, eps, n_actions:np.int32, batch_size,
                 replace, input_dims, eps_dec=1e-5, eps_min=0.01, mem_size=1000000):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.eps = eps
        self.batch_size = batch_size
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.replace = replace          # how often we update target q network weights
        self.input_dims = input_dims
        self.lr = lr
        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        self.Q_target = DQN(input_dims, n_actions, lr)
        self.Q_eval = DQN(input_dims, n_actions, lr)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_eval.to(device)
        self.Q_target.to(device)

    def replace_target_network(self):
        if self.replace != 0 and self.learn_step % self.replace == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = torch.tensor([observation]).to(device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        self.learn_step += 1
        if self.memory.pointer < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        state, action_batch, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        self.replace_target_network()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(state).to(device)
        new_state_batch = torch.tensor(new_state).to(device)
        reward_batch = torch.tensor(reward).to(device)
        terminal_batch = torch.tensor(done).to(device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min
