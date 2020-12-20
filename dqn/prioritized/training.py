import time
import plotly.graph_objects as go
from IPython.display import clear_output

from ..utils import last_rolling_mean

class PTraining():

    def __init__(self, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 beta_start=0.4, beta_inc=1.002, max_t=1000):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = self.eps_start
        self.beta = beta_start
        self.beta_inc = beta_inc

    def play(self, agent, env, brain_name):
        self.brain_name = brain_name
        scores = []
        for i in range(5):
            score = self.__play_episode(agent, env, train=False)
            scores.append(score)
        return scores

    def train(self, agent, env, brain_name, success_thresh=13.0,
              track_every=100, plot=False, weights='checkpoint.pth'):
        self.brain_name = brain_name
        scores = []
        rolling = []
        fig = go.Figure(layout=dict(height=350, xaxis_title='# Episode',
                        yaxis_title='Score')) if plot else None
        for i_episode in range(1, self.n_episodes+1):
            score = self.__play_episode(agent, env)
            scores.append(score)
            rolling.append(last_rolling_mean(scores))
            self.__track_results(i_episode, scores, rolling, track_every, fig)
            if rolling[-1] > success_thresh:
                self.__terminate(agent, i_episode, rolling, weights)
                break 
        return scores

    def __update_params(self):
        self.eps = max(self.eps_end, self.eps_decay*self.eps)
        self.beta = min(1.0, self.beta_inc*self.beta)

    def __interact(self, agent, state, env, train=True):
        if train:
            action = agent.act(state, self.eps)
        else:
            action = agent.act(state)
        env_info = env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return action, reward, next_state, done

    def __play_episode(self, agent, env, train=True):
        state = self.__reset(env)
        score = 0                            
        for t in range(self.max_t):
            action, reward, next_state, done = self.__interact(agent, state, env, train)
            if train:
                agent.step(state, action, reward, next_state, done, self.beta)
            else:
                time.sleep(0.045)
            score += reward
            state = next_state
            if done:
                break
        self.__update_params()
        return score

    def __reset(self, env):
        env_info = env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations[0]

    def __track_results(self, i_episode, scores, rolling, track_every, fig=None):
        if fig is None:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, rolling[-1]), end="")
            if i_episode % track_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, rolling[-1])) 
        elif i_episode % track_every == 0:
            fig.update(data=[go.Scatter(y=scores, line_color='#2664A5', opacity=0.5, name='scores'),
                             go.Scatter(y=rolling, line_color='#2664A5', name='running mean')])
            fig.update_layout(title='Running mean at episode {}: {}'.format(i_episode,
                                                                           rolling[-1]))
            fig.show() 
            clear_output(wait=True)

    def __terminate(self, agent, i_episode, rolling, weights):
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100,
                                                                                     rolling[-1]))
        agent.save_weights(weights)
