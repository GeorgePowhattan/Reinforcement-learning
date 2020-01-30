# Initial_population - to generate randomly successful training data together with actions that led to them to ??? train the NN ???

import gym
import numpy as np
import random
from statistics import median, mean
from collections import Counter

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def initial_population():
    
    training_data = []        # [OBSERVATION, ACTION]
    scores = []
    accepted_scores = []      # just the scores that met our threshold:
    
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in max 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)
            
            # notice that the observation is returned FROM the action so we'll store the previous observation here, pairing the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save every move we made.
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying to influence the machine in any way as to HOW that score is reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:     # if action was 1
                    output = [0,1]
                elif data[1] == 0:   # if action was 0
                    output = [1,0]
                    
                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    print(np.array(training_data))
    
    return training_data