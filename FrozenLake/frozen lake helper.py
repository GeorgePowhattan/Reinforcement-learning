import gym

env = gym.make('FrozenLake-v0')
env.reset()

action_space_size = env.action_space.n
obs_space_size = env.observation_space.n
#print(type(state_max))
#print(type(state_min))

print('\n*********** Basic Action/Observation space type **********')
print('Action space: ', env.action_space)
print('Observ space: ', env.observation_space)

print('\n*********** Action space parameters **********')
print('Action space size: ', action_space_size)
#print('Action space high: ',env.action_space.high)
#print('Action space low: ',env.action_space.low)
print('Action space sample: ', [env.action_space.sample() for _ in range (10)] )

print('\n*********** Observation space parameters **********')
print('Observation space size: ', obs_space_size)
#print('Observation space high: ',env.observation_space.high)
#print('Observation space low: ',env.observation_space.low)
print('Observation space sample: ', [env.observation_space.sample() for _ in range (10)] )


'''The DISCRETE space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1. The BOX space represents an n-dimensional box, so valid observations will be an array of 4 numbers.'''

print('\n*********** Reward parameters **********')
print('Reward range: ', env.reward_range)
# print(env.goal_position)


env.close()

