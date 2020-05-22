import torch
from base_dqns import DQN, EpsilonAgent
import numpy as np
import bsuite

if __name__ == '__main__':
    env = bsuite.load_from_id('deep_sea/0')
    num_actions = env.action_spec().num_values

    agent = EpsilonAgent(gamma=0.99, eps=1.0, lr=0.003, input_dims=100, output_dims=2,
                         batch_size=128, n_actions=2, max_mem_size=100000, eps_end=0.01, eps_dec=1e-4)

    scores = []
    eps_history = []
    episodes = 500
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

        fmt = 'episode {}, score {:.2f}, avg_score {:.2f}, eps {:.3f}'
        print(fmt.format(i, score, avg_score, agent.eps))
