import numpy as np
import gym

env = gym.make("MountainCar-v0")

action_space_size = env.action_space.n
state_max = env.observation_space.high
state_min = env.observation_space.low

print(state_max)
print(state_min)

print(type(state_max))
print(type(state_min))

discrete_os_size = [20,20]
discrete_os_win_size = (state_max - state_min) / discrete_os_size

print(discrete_os_win_size)

env.close()