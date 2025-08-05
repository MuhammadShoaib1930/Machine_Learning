


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
alpha = 0.1
gamma = 0.9
epsilon = 0.1

for episode in range(1000):
    state = env.reset()
    action = np.argmax(q_table[state]) if random.uniform(0,1) > epsilon else env.action_space.sample()
    done = False

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = np.argmax(q_table[next_state]) if random.uniform(0,1) > epsilon else env.action_space.sample()

        q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])
        state, action = next_state, next_action
