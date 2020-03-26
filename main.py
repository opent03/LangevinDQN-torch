import gym
import torch
from convdqn import Agent
import numpy as np
from skimage import color
from utils import make_env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4', n_frames=4)
    agent = Agent(gamma=0.95, eps=1.0, lr=1e-4, input_dims=(4,80,80),
     batch_size=32, n_actions=6, max_mem_size=25000, eps_end=0.02, eps_dec=1e-5)

    scores = []
    eps_history = []
    episodes = 250
    score = 0
    for i in range(episodes):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):i+1])
            print('episode {}, score {},'
                  ' average score {}, epsilon {:.3f}'.format(i, score, avg_score, agent.eps))
        else:
            print('episode {}, score {}, epsilon {:.3f}'.format(i, score, agent.eps))
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
    torch.save(agent.Q_eval.state_dict(), './saves/')