import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DQN(nn.Module):
    def __init__(self, alpha):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 4)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        x = torch.tensor(obs).to(self.device)
        x = x.view(-1, 1, 185, 95)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        actions = self.fc2(x)

        return actions

class Agent:
    def __init__(self, gamma, eps, alpha, mem_size, eps_end = 0.05, replace=10000, action_space=[0,1,2,3]):
        self.gamma = gamma
        self.eps = eps
        self.eps_end = eps_end
        self.action_space = action_space
        self.mem_size = mem_size
        self.memory = []
        self.steps = 0
        self.learn_step_counter = 0             # number of times agent calls learn
        self.mem_pointer = 0                    # position in memory
        self.replace_target_count = replace
        self.Q_eval = DQN(alpha)
        self.Q_target = DQN(alpha)

    def store_transition(self, state, action, reward, next_state):
        if self.mem_pointer < self.mem_size:
            self.memory.append([state, action, reward, next_state])
        else:
            self.memory[self.mem_pointer%self.mem_size] = [state, action, reward, next_state]
        self.mem_pointer += 1

    def eps_greedy_choice(self, obs):
        # this is actually not really needed, the Langevin SGD steps is itself an exploration scheme
        rand = np.random.random()
        actions = self.Q_eval.forward(obs)
        if rand < 1-self.eps:
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        # fixed target network
        if self.replace_target_count is not None and self.learn_step_counter % self.replace_target_count == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        # i hate this
        if self.mem_pointer + batch_size < self.mem_size:
            mem_start = int(np.random.choice(range(self.mem_pointer)))
        else:
            mem_start = int(np.random.choice(range(self.mem_pointer-batch_size-1)))
        batch = self.memory[mem_start:mem_start + batch_size]
        batch = np.array(batch)

        # get Q-values
        Qpred = self.Q_eval.forward(list(batch[:,0])).to(device)
        Qnext = self.Q_target.forward(list(batch[:,3])).to(device)

        max_actions = torch.argmax(Qnext, dim=1).to(device)
        rewards = torch.tensor(list(batch[:,2])).to(device)
        Qtarget = Qpred
        # make td target
        Qtarget[:,max_actions] = rewards + self.gamma*torch.max(Qnext[1])

        # decay eps
        if self.steps > 1000:
            if self.eps - 1e-4 > self.eps_end:
                self.eps -= 1e-4
            else:
                self.eps = self.eps_end

        loss = self.Q_eval.loss(Qtarget, Qpred).to(device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = Agent(gamma=0.99, eps=0.95, alpha=3e-3, mem_size=5000)

    # fill up memory first
    while agent.mem_pointer < agent.mem_size:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs_next, reward, done, info = env.step(action)
            agent.store_transition(np.mean(obs, axis=2), action, reward, obs_next)
            obs = obs_next

    print("Done initialization!")

    scores = []
    epsilons = []
    episodes = 40
    batch_size = 32

    for episode in range(episodes):
        print('Starting game ', episode+1, 'epsilon: {:.4f}'.format(agent.eps))
        epsilons.append(agent.eps)
        done = False
        obs = env.reset()
        frames = [np.sum(obs, axis=2)]
        score = 0
        last_action = 0

        while not done:
            if len(frames) == 3:
                action = agent.eps_greedy_choice(frames)
                frames = []
            else:
                action = last_action
            obs_next, reward, done, info = env.step(action)
            print(obs_next.shape)
            score += reward
            frames.append(np.sum(obs, axis=2))
            agent.store_transition(np.mean(obs, axis=2), action, reward, obs_next)
            obs = obs_next

            agent.learn(batch_size)
            last_action = action
            env.render()
        scores.append(score)
        print('score: ', score)