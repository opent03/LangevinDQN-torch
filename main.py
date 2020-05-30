import gym
import os
import torch
from base_dqns import Agent
import numpy as np
from skimage import color
from utils import make_env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.envs.make('CartPole-v1')
    agent = Agent(gamma=0.99, eps=1.0, lr=1e-4, input_dims=4, output_dims=2,
     batch_size=128, n_actions=2, max_mem_size=100000, eps_end=0.02, eps_dec=1e-4, langevin=True)

    scores = []
    avg_scores = []
    eps_history = []
    episodes = 5000
    score = 0
    for episode in range(episodes):
        if episode % 5 == 0:
            agent.replace_target_network()
        score = 0
        eps_history.append(agent.eps)
        #observation = img_preprocessing(env.reset())
        observation = env.reset()
        # sequence of frames
        #frames = [observation]
        # = 0

        done = False
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #observation_ = img_preprocessing(observation_)
            score += reward
            #if done and info['ale.lives'] == 0:
            #    reward = -100

            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            last_action = action

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        fmt = 'episode {}, score {:.2f}, avg_score {:.2f}, eps {:.4f}'
        print(fmt.format(episode + 1, score, avg_score, agent.eps))

    # save results
    path = 'saves/'
    scores = np.array(scores)
    avg_scores = np.array(avg_scores)
    np.save(os.path.join(path, '{}_scores.npy'.format('langevin' if agent.langevin else 'adam')), scores)
    np.save(os.path.join(path, '{}_avg_scores.npy'.format('langevin' if agent.langevin else 'adam')), avg_scores)