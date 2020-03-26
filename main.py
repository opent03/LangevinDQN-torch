import gym
import torch
from DQN_breakout import Agent
import numpy as np
from skimage import color

def img_preprocessing(image):
    image = image[::2,::2,]
    image = color.rgb2gray(image)[:, :, np.newaxis]
    return np.transpose(image, (2,0,1))

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = Agent( gamma=0.99, epsilon=1.0, lr=0.003, in_dim=1, out_dim=4, batch_size=64,
                 mem_capacity=10000, eps_min=0.01, eps_dec=0.999)
    scores = []
    eps_history = []
    episodes = 500
    score = 0
    for i in range(episodes):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):i+1])
            print('episode {}, score {},'
                  ' average score {}, epsilon {:.3f}'.format(i, score, avg_score, agent.epsilon))
        else:
            print('episode {}, score {}, epsilon {:.3f}'.format(i, score, agent.epsilon))
        score = 0
        eps_history.append(agent.epsilon)
        observation = img_preprocessing(env.reset())

        # sequence of frames
        frames = [observation]
        last_action = 0

        done = False
        while not done:
            env.render()
            if len(frames) == 3:
                action = agent.choose_action(np.array(frames))
                frames = []
            else:
                action = last_action
            try:
                observation_, reward, done, info = env.step(action)
            except:
                print(last_action)
                print(action)
                exit(1)
            observation_ = img_preprocessing(observation_)
            score += reward
            frames.append(observation_)
            agent.memory.push(observation, action, reward, observation_)
            agent.learn()
            observation = observation_
            last_action = action

        scores.append(score)
    torch.save(agent.Q_eval.state_dict(), './saves/')