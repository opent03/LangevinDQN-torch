import gym
from collections import namedtuple
import time
import numpy as np
env = gym.make('Pong-v0')
obs = env.reset()
print(env.action_space)
print(env.observation_space)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
t = [Transition(np.array([2,3]),3,4,5), Transition(np.array([2,3]),7,8,9)]
batch = Transition(*zip(*t))
batch_state = np.asarray(batch.state)
print(batch_state, type(batch_state))

exit(0)
for i in range(10000):
    env.render()
    time.sleep(0.05)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print("Done")
        break

env.close()