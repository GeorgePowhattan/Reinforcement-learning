import numpy as np
import gym
#from gym.envs import toy_text
#toy_text.frozen_lake

import random
import time

#env = gym.make('MountainCar-v0')
env = gym.make("Blackjack-v0")
env.reset()
env.render()

#action_space_size = env.action_space.n
#state_space_size = env.observation_space.n

#q_table = np.zeros((state_space_size, action_space_size))

#print(action_space_size)

env.close()