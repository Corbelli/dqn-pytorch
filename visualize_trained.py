import gym
import torch

from dqn_agent import Agent


def random(evn, n_episodes=3):
    agent = Agent(state_size=8, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(200):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    random(env)