from matplotlib import pyplot as plt
import numpy as np
import argparse

def load_data(env='', langevin=True):
    fmt = 'langevin' if langevin else 'adam'
    scores = np.load('{}_scores_{}.npy'.format(fmt, env))
    #scores = np.load('langevin_adam.npy')
    avg_scores = np.load('{}_avg_scores_{}.npy'.format(fmt, env))
    return scores, avg_scores

def plot_data(scores, avg_scores, env):
    plt.figure()

    x = np.arange(len(scores))
    plt.plot(x, scores)
    plt.plot(x, avg_scores)
    plt.xlabel('episode')
    plt.ylabel('score/avg_score')
    plt.title('{} environment: scores against episodes'.format(env))
    #plt.figure()
    #mva_scores = moving_average(scores, 20)
    #plt.plot(np.arange(len(mva_scores)), mva_scores)
    plt.show()

def moving_average(array, window=10):
    mva = []
    for i in range(len(array)-window):
        mva.append(np.sum(array[i:i+window])/window)
    return np.array(mva)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='deepsea', type=str)
    parser.add_argument('--langevin', default=False, type=bool)
    args = parser.parse_args()
    scores, avg_scores = load_data(env=args.env, langevin=args.langevin)
    plot_data(scores, avg_scores, args.env)