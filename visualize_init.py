import gym


def random(evn, n_episodes=3):
    for i_episode in range(n_episodes):
        observation = env.reset()
        for t in range(200):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    random(env)