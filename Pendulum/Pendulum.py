import numpy as np
import gym
import random

env = gym.make("Pendulum-v0")

# Hyperparameters:
epsilon = 1 # the higher, the more exploratory
min_epsilon = 0.01
max_epsilon = 1
epsilon_decay = 0.002

episodes = 400
max_steps_per_turn = 500
learning_rate = 0.1
discount_rate = 0.95

# Action space ---------------------------------------------------------------
discrete_act_size = 40
action_space_size = (env.action_space.high - env.action_space.low) / discrete_act_size

def get_discrete_action_state(action):
    discrete_action = (action - env.action_space.low)/discrete_act_size
    return tuple(discrete_action.astype(np.int))

# State space ----------------------------------------------------------------
discrete_os_size = [40,40,40]
discrete_os_win_size = (np.array(env.observation_space.high) - np.array(env.observation_space.low) ) / discrete_os_size

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [discrete_act_size]))

# -----------------------------------------------------------------------------
# TRAINING --------------------------------------------------------------------
# -----------------------------------------------------------------------------
for episode in range(episodes):
    print(episode)
    rewards_current_episode = 0
    discrete_state = get_discrete_state(env.reset())

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay*episode)

    # render only xth episode
    if episode % 100 == 0:
        render = True
        print(f"{episode}th training reached")
    else:
        render = False
    
    done = False 

    for i in range(max_steps_per_turn): 

        dice = random.uniform(0,1)
        # action
        if dice > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()

        discrete_action = get_discrete_action_state(action)
        new_state, reward, done, _ = env.step(discrete_action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % 100 == 0 and render == True:
            env.render()   

        q_table[discrete_state + (discrete_action,)] = q_table[discrete_state + (discrete_action,)] * (1 - learning_rate)  +  learning_rate * (reward + discount_rate * np.max(q_table[new_discrete_state]))

        if np.array_equal(new_state,np.array([0,1,0])):
            print(f"We have reached our goal in episode {episode}")
        
            q_table[discrete_state + (discrete_action,)] = 0
            
            break
        
        discrete_state = new_discrete_state

env.close()