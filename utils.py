import numpy as np
import gym
import os

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

class SkipEnv(gym.Wrapper):
    ' Skips over some frames '
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward =0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

class PreprocessFrame(gym.ObservationWrapper):
    ' Process 4 frames at a time '
    def __init__(self, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80,80,1), dtype=np.uint8)

    def observation(self, obs):
        return PreprocessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + 0.114*new_frame[:,:,2]
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)

        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
            shape=(self.observation_space.shape[-1],
                   self.observation_space.shape[0],
                   self.observation_space.shape[1]),
                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2,0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper,self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer

def make_env(env_name, n_frames):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreprocessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, n_frames)
    return ScaleFrame(env)


def save_results(path, agent, env, scores, avg_scores):
    scores = np.array(scores)
    avg_scores = np.array(avg_scores)
    np.save(os.path.join(path, '{}_scores_{}.npy'.format('langevin' if agent.langevin else 'adam',env)), scores)
    np.save(os.path.join(path, '{}_avg_scores_{}.npy'.format('langevin' if agent.langevin else 'adam', env)),
            avg_scores)