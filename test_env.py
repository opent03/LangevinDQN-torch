import gym
import time
env = gym.make('Breakout-v0')
obs = env.reset()
print(env.action_space)
print(env.observation_space)
for i in range(10000):
    env.render()
    time.sleep(0.05)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print("Done")
        break

env.close()