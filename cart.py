import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPISODES = 10000  #how many iterations of the game we'd like to run.

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# returns tuple (2x1)
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(2000):
    discrete_state = get_discrete_state(env.reset())
    print(discrete_state)

    done = False

    if episode % 400 == 0:
        render = True
        print(f"{episode}th training reached")
    else:
        render = False


    while not done:
        action = np.argmax(q_table[discrete_state]) # action - needs to be the best choice in the given situation, return the action for the given combination of state tuple
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        if episode % 400 == 0 and render == True:
            env.render()   

        # If simulation did not end yet after last step - UPDATE Q TABLE
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)] # returns Q for (3x1)

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q  +  LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # if goal is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            print(f"We have reached our goal in episode {episode}")
            
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()