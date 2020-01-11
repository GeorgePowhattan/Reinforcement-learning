import numpy as np
import gym
import random

env = gym.make('FrozenLake-v0')
env.reset()

# Hyperparameters: Epsilon
epsilon = 1 # the higher, the more exploratory
min_epsilon = 0.01
max_epsilon = 1
epsilon_decay = 0.001
   
episodes = 20000
max_steps_per_turn = 100
learning_rate = 0.1
discount_rate = 0.98
rewards_all_episodes = []

# Action space ---------------------------------------------------------------
action_space_size = env.action_space.n

# State space ----------------------------------------------------------------
obs_space_size = env.observation_space.n

q_table = np.random.uniform(low=-2, high=0, size=(obs_space_size, action_space_size))


# -----------------------------------------------------------------------------
# TRAINING --------------------------------------------------------------------
# -----------------------------------------------------------------------------
for episode in range(episodes):

    print(episode)
    discrete_state = env.reset()

    # Exploration-exploitation tradeoff
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay*episode)

    # Render environment if Xth episode is reached
    '''if episode % 1000 == 0:
        render = True
        print(f"{episode}th training reached")
    else:
        render = False
    '''
    done = False 


    for step in range(max_steps_per_turn):
        
        # Chose action:
        dice = random.uniform(0,1)
        if dice > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()
        
        # Render environment if Xth episode is reached
        #if episode % 10 == 0 and render == True:
        #    env.render() 

        # Take action
        new_state, reward, done, _ = env.step(action)
        
        # Update Q-table:
        q_table[discrete_state, action] = q_table[discrete_state, action] * (1 - learning_rate)  +  learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Set new state:
        discrete_state = new_state

        if done:
            
            if reward == 1:
                print(f"We have reached our goal in episode {episode}")
                # Add reward to rewards
                rewards_all_episodes.append(reward)
                break
            
            else:
                print(f"Fallen to ice in episode {episode}")
                rewards_all_episodes.append(0)
                break

print("sum of all rewards: {}".format(sum(rewards_all_episodes)))
print("episodes: {}".format(episodes))
print("length of rewards_all_episodes: {}".format(len(rewards_all_episodes)))

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), episodes/1000)
count = 1000

print("\n********Average reward per 1000 episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", sum(r)/1000)
    count += 1000

env.close()
