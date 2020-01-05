import numpy as np
import gym

env = gym.make("Pendulum-v0")
#env.reset()
#env.render()

action_space_size = env.action_space
obs_space_size = env.observation_space

#q_table = np.zeros((state_space_size, action_space_size))

print(action_space_size)
print(obs_space_size)

print(env.action_space.high)
print(env.action_space.low)

#print(env.action_space.high)
#print(env.action_space.low)

print(env.observation_space.high)
print(env.observation_space.low)

print(type(env.observation_space.high))
print(type(env.observation_space.low))

# print(env.goal_position)
print(env.reward_range)


env.close()