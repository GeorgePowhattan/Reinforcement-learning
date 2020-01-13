import numpy as np
import random
import tensorflow

episodes = 1200

#rewards_all_episodes = [20,40,60,80,100]
rewards_all_episodes = np.random.randint(0,2,episodes)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), episodes/400)
count = 400

print("********Average reward per 400 episodes********\n")
for r in rewards_per_thousand_episodes:
    print(r)
    print(count, ": ", sum(r/400))
    count += 400

