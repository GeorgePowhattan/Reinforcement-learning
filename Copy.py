#####
# Monte Carlo
#####
import gym
#from gym.wrappers.monitoring import Monitor
from random import randint
import numpy as np

def learn(episodeCount):
	for i_episode in range(episodeCount):
		obs = env.reset()
		episodeStatesActions = []
		totalRewards = 0
		for t in range(200):
			action = policy(obs, i_episode)
			obs, reward, done, _ = env.step(action['action'])
			episodeStatesActions.append(action)
			totalRewards += reward
			if done:
				for i, a in enumerate(episodeStatesActions):
					updatePolicy(a, totalRewards-i)
				break


def actionExists(obs, action):
	for a in history[obs]:
		if actionAreEqual(a['action'], action['action']):
			return True
	return False


def actionAreEqual(action1, action2):
	return action1[0] == action2[0] and action1[1] == action2[1] and action1[2] == action2[2]


def updatePolicy(action, G):
	action['value'] = (action['value'] * action['count'] + G) / (action['count'] + 1)
	action['count'] += 1


def policy(obs, i_episode):
	# Create obs in history if it does not exist
	if obs not in history: history[obs] = []
	# Get the most valued action
	maxValueAction = None
	for a in history[obs]:
		if maxValueAction == None or a['value'] > maxValueAction['value']:
			maxValueAction = a
	# If no maxValueAction or if it's time for exploration, return random action
	if maxValueAction == None or randint(0, 100) < 20*0.99**i_episode:
		randomAction = {'value': 0, 'action': env.action_space.sample(), 'count': 0}
		if not actionExists(obs, randomAction):
			history[obs].append(randomAction)
		return randomAction
	else:
		return maxValueAction


nbEpisodes = 100000

env = gym.make('Copy-v0')
#env = Monitor(env, 'tmp/copy', force=True)

history = {}
learn(nbEpisodes)

env.close()