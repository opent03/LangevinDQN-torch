import torch
import os
from base_dqns import Agent
import numpy as np
import bsuite
from utils import save_results



if __name__ == '__main__':
    path = 'saves/'
    env = bsuite.load_from_id('deep_sea/0')
    num_actions = env.action_spec().num_values

    agent = Agent(gamma=0.99, eps=1.0, lr=(0.0002*0.0001), input_dims=100, output_dims=2,
                         batch_size=128, n_actions=2, max_mem_size=100000, eps_end=0.01, eps_dec=1e-4, langevin=True)

    scores = []
    avg_scores = []
    eps_history = []
    episodes = 10000
    try:
        for i in range(episodes):
            score = 0
            eps_history.append(agent.eps)
            timestep = env.reset()
            while not timestep.last():
                observation = timestep.observation
                observation = np.reshape(observation, (-1))
                action = agent.choose_action(observation)
                timestep_ = env.step(action)

                observation_, reward, done = timestep_.observation, timestep_.reward, timestep_.last()
                # observation_ = img_preprocessing(observation_)
                score += reward
                # if done and info['ale.lives'] == 0:
                #    reward = -100
                observation_ = np.reshape(observation_, (-1))
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                #observation = observation_
                timestep = timestep_
                last_action = action

            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            fmt = 'episode {}, score {:.2f}, avg_score {:.2f}, eps {:.3f}'
            print(fmt.format(i, score, avg_score, agent.eps))
        save_results(path, agent, 'deepsea', scores, avg_scores)
    except KeyboardInterrupt:
        if input('Save results? (Y/N, default Y)') != 'N':
            # save results/
            print('Saving...')
            save_results(path, agent, 'deepsea', scores, avg_scores)
        else:
            print('Not saving...')



