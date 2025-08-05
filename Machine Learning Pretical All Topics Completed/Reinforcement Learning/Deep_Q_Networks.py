
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

model = keras.Sequential([
    keras.layers.Dense(24, input_shape=(env.observation_space.shape[0],), activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
