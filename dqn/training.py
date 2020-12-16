import torch
from collections import deque
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

COLORS = [
     '#3195DA',
     '#FF0063',
     '#0C1425',
     '#57D493'
 ]

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

    def interact(self, agent, state, env, train=True):
        if train:
            action = agent.act(state, self.eps)
        else:
            action = agent.act(state)
        env_info = env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return action, reward, next_state, done

    def play_episode(self, agent, env, train=True):
        state = self.reset(env)
        score = 0                            
        for t in range(self.max_t):
            action, reward, next_state, done = self.interact(agent, state, env, train)
            if train:
                agent.step(state, action, reward, next_state, done)
            else:
                time.sleep(0.045)
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
        fig = go.Figure(layout=dict(title='Training of Model', height=350,
                        xaxis_title='# Episode', yaxis_title='Score')) if plot else None
        for i_episode in range(1, self.n_episodes+1):
            score = self.play_episode(agent, env)
            scores.append(score)
            scores_window.append(score)
            self.track_results(agent, i_episode, scores, scores_window, track_every, fig)
            if np.mean(scores_window) > success_thresh:
                self.terminate(agent, i_episode, scores, scores_window)
                break 
        return scores

    def track_results(self, agent, i_episode, scores, scores_window, track_every, fig=None):
        if fig is None:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % track_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window))) 
        elif i_episode % track_every == 0:
            fig.update(data=[go.Scatter(y=scores, line_color='#2664A5')])
            fig.show() 
            clear_output(wait=True)

    def terminate(self, agent, i_episode, scores, scores_window):
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')

    def play(self, agent, env, brain_name):
        self.brain_name = brain_name
        scores = []
        for i in range(5):
            score = self.play_episode(agent, env, train=False)
            scores.append(score)
        return scores


class Benchmarks():

    def __init__(self):
        pass

    def save_score(self, name, scores, description):
        with open("benchmarks/{}_score.txt".format(name), "wb") as fp:
            pickle.dump(scores, fp)
        with open("benchmarks/{}_description.txt".format(name), "wb") as fp:
            pickle.dump(description, fp)

    def description(self, name):
        with open("benchmarks/{}_description.txt".format(name), "rb") as fp:
            description = pickle.load(fp)
        return description

    def load_benchmarks(self):
        files = [f.split('_score')[0] for f in listdir('benchmarks/') if isfile(join('benchmarks/', f)) and 'score' in f]
        return {f: self.load(f) for f in files} 

    def load(self, name):
        with open("benchmarks/{}_score.txt".format(name), "rb") as fp:
            scores = pickle.load(fp)
        return scores

    def plot_bench(self, scores, title='', mean=None, opacity=0.5):
        if mean is None:
            fig = go.Figure(data=go.Scatter(y=scores, line_color=COLORS[0]))
        else:
            fig = go.Figure(data=go.Scatter(y=scores, line_color=COLORS[0], opacity=opacity, showlegend=False))
            means = calc_rollling_mean(scores, mean)
            fig.add_trace(go.Scatter(y=means, line_color=COLORS[0], showlegend=False))
        fig.update_layout(title=title,
                          xaxis_title='Episode', yaxis_title='Episode Reward')
        fig.show() 
        return fig

    def plot_benchs(self, bench_dict, title='', mean=None, opacity=0.5):
        fig = go.Figure()
        for i, (name, scores) in enumerate(bench_dict.items()):
            if mean is None:
                fig.add_trace(go.Scatter(y=scores, name=name, showlegend=True, line_color=COLORS[i]))
            else:
                fig.add_trace(go.Scatter(y=scores, name=name, opacity=opacity, showlegend=True, line_color=COLORS[i]))
                means = calc_rollling_mean(scores, mean)
                fig.add_trace(go.Scatter(y=means, showlegend=False, line_color=COLORS[i]))
        fig.update_layout(title=title,
                    xaxis_title='# Episode', yaxis_title='Episode Reward')
        fig.show()

def calc_rollling_mean(v, k=100):
 return [np.mean(v[(i-k+1):(i+1)]) if i-k+1 >= 0 else np.mean(v[0:(i+1)]) for i in range(len(v))]