import torch
from collections import deque
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class Training():

    def __init__(self, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_t=1000):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = self.eps_start

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps_decay*self.eps)

    def interact(self, agent, state, env):
        action = agent.act(state, self.eps)
        env_info = env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return action, reward, next_state, done

    def play_episode(self, agent, env):
        state = self.reset(env)
        score = 0                            
        for t in range(self.max_t):
            action, reward, next_state, done = self.interact(agent, state, env)
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        self.decay_eps()
        return score

    def reset(self, env):
        env_info = env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations[0]

    def train(self, agent, env, brain_name, success_thresh=13.0, track_every=100, plot=False):
        self.brain_name = brain_name
        scores = []
        scores_window = deque(maxlen=100)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1) 
        for i_episode in range(1, self.n_episodes+1):
            score = self.play_episode(agent, env)
            scores.append(score)
            scores_window.append(score)
            self.track_results(agent, i_episode, scores, scores_window, track_every, fig, ax, plot)
            if np.mean(scores_window) > success_thresh:
                self.terminate(agent, i_episode, scores, scores_window)
                break 
        return scores

    def track_results(self, agent, i_episode, scores, scores_window, track_every, fig, ax, plot):
        if not plot:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % track_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window))) 
        elif i_episode % track_every == 0:
            ax.set_xlim(0, self.n_episodes)
            ax.set_ylim(-5, 25)
            ax.plot(scores, color='b', linewidth=1, label='episode score')
            ax.set_title('Average at episode {:d}: {:.2f}'.format(i_episode, np.mean(scores_window)))
            display(fig)
            clear_output(wait = True)

    def terminate(self, agent, i_episode, scores, scores_window):
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')


class Benchmarks():

    def __init__(self):
        pass

    def save_score(self, name, scores, description):
        with open("benchmarks/{}_score.txt".format(name), "wb") as fp:
            pickle.dump(scores, fp)

    def description(self, name):
        with open("benchmarks/{}_description.txt".format(name), "wb") as fp:
            pickle.dump(scores, fp)

    def load_benchmarks(self):
        onlyfiles = [f for f in listdir('benchmarks/') if isfile(join('benchmarks/', f)) and 'score' in f]
        scores = []
        for f in onlyfiles:
            with open("benchmarks/{}".format(f), "wb") as fp:
                scores.append(pickle.dump(scores, fp))
        return scores

    def load(self, name):
        with open("benchmarks/{}_score.txt".format(name), "rb") as fp:
            scores = pickle.load(fp)
        return scores

    def plot_bench(self, scores, name=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title(name)
        plt.show()
        pass

    def plot_benchs(self, bench_dict, name=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for name, scores in bench_dict.items():
            plt.plot(np.arange(len(scores)), scores, label='name')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title(name)
        plt.show()
        pass