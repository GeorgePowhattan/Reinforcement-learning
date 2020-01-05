import numpy as np
import gym
import random
import time

env = gym.make("MountainCar-v0")

# Hyperparameters: Epsilon
epsilon = 1 # the higher, the more exploratory
min_epsilon = 0.01
max_epsilon = 1
epsilon_decay = 0.002
   
episodes = 2000
max_steps_per_turn = 500
learning_rate = 0.1
discount_rate = 0.95
rewards_all_episodes=[]

action_space_size = env.action_space.n
state_max = env.observation_space.high
state_min = env.observation_space.low

discrete_os_size = [20,20]
discrete_os_win_size = (state_max - state_min) / discrete_os_size

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

#Helper function to discretise, returns shape(2,1)
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# Episodes loop
for episode in range(episodes):
    rewards_current_episode = 0
    discrete_state = get_discrete_state(env.reset())
    
    # epsilon linear decay
    #epsilon = epsilon - epsilon * (episode/episodes)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay*episode)

    # render only xth episode
    if episode % 400 == 0:
        render = True
        print(f"{episode}th training reached")
    else:
        render = False
    
    done = False 
    rewards_all_episodes.append(0)

    for i in range(max_steps_per_turn): 
        
        # exploration vs. exploitation:
        dice = random.uniform(0,1)
        # action
        if dice > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % 400 == 0 and render == True:
            env.render()           
        
        # update Q-table
        q_table[discrete_state + (action,)] = q_table[discrete_state + (action,)] * (1 - learning_rate)  +  learning_rate * (reward + discount_rate * np.max(q_table[new_discrete_state]))


        if new_state[0] >= env.goal_position:
            print(f"We have reached our goal in episode {episode}")
            q_table[discrete_state + (action,)] = 0
            # reward object not working dor this continuous environment -> we must manually add 1 to last "rewards_all_episodes" item
            rewards_all_episodes[-1] += 1
            
            break
        
        discrete_state = new_discrete_state


# After all episodes are over, calculate avrg reward per 400 episodes:
print("sum of all rewards: {}".format(sum(rewards_all_episodes)))
print("episodes: {}".format(episodes))
print("length of rewards_all_episodes: {}".format(len(rewards_all_episodes)))

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), episodes/400)
count = 400

print("********Average reward per 400 episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", sum(r)/400)
    count += 400

env.close()