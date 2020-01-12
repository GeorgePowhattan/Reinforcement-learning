import numpy as np
import gym

env = gym.make("MountainCar-v0")
env.reset()

discrete_os_size = [20,20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

print('\n*********** Basic Action/Observation space type **********')
print('Action space: ', env.action_space)
print('Observ space: ', env.observation_space)
print('Goal position: ', env.goal_position())

print('\n*********** Action space parameters **********')
print('Action space size: ', env.action_space.n)
#print('Action space high: ',env.action_space.high)
#print('Action space low: ',env.action_space.low)
print('Action space sample: ', [env.action_space.sample() for _ in range (10)] )

print('\n*********** Observation space parameters **********')
print('Observation space size: ', env.observation_space)
print('Observation space high: ',env.observation_space.high)
print('Observation space low: ',env.observation_space.low)
#print('Observation space sample: ', [env.observation_space.sample() for _ in range (10)] )


'''The DISCRETE space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1. The BOX space represents an n-dimensional box, so valid observations will be an array of 4 numbers.'''

print('\n*********** Reward parameters **********')
print('Reward range: ', env.reward_range)
# print(env.goal_position)

env.close()