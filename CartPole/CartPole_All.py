import gym
import random
import numpy as np

from Model import neural_network_model, train_model
from Initial_population import initial_population

env = gym.make("CartPole-v0")
env.reset()

# Hyperparameters:
goal_steps = 500
score_requirement = 50
initial_games = 10000

scores = []
choices = []

if __name__ == "__main__":
    
    # filter initial training data through randomly good games
    training_data = initial_population()
    # train NN based on the training data
    model = train_model(training_data)

    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()
            
            if len(prev_obs)==0:   # the first action is random
                action = random.randrange(0,2)
            else:
                action = np.argmax( model.predict(prev_obs.reshape(-1,len(prev_obs),1)) [0] )

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)
