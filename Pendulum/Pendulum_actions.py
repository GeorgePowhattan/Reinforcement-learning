import numpy as np
import gym

env = gym.make("Pendulum-v0")
#env.reset()
#env.render()

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(reward)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()