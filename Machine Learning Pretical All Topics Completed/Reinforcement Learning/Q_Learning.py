# Common libraries
import gym            # For environments
import numpy as np    # For arrays and math
import random         # For randomness
import matplotlib.pyplot as plt  # For plotting

# For Deep RL
import tensorflow as tf
from tensorflow import keras
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
state = env.reset()
print("Initial State:", state)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1   # Learning rate
gamma = 0.6   # Discount factor
epsilon = 0.1 # Exploration rate

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        # Choose action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update Q-value
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
