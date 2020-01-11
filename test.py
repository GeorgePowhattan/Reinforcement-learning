import matplotlib.pyplot as plt
import numpy as np

min_epsilon = 0
max_epsilon = 1
epsilon_decay = 0.01
epsilon = []

for episode in range(1200):
      
    # epsilon linear decay
    #epsilon = epsilon - epsilon * (episode/episodes)
    eps = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay*episode)
    
    epsilon.append(eps)

# plot the epsilon:

plt.plot(range(1200), epsilon)
plt.show()

'''
x = np.linspace(-np.pi, np.pi, 201)
plt.plot(x, np.sin(x))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()'''