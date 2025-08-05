# !pip install pymdptoolbox
#  Markov Decision Process (MDP)
import mdptoolbox
import mdptoolbox.example
import numpy as np

# Transition matrix: a list of 2 matrices (one per action)
P = [
    np.array([
        [0.5, 0.5, 0.0],  # action 0, from state 0 to states 0,1,2
        [0.2, 0.3, 0.5],  # action 0, from state 1
        [0.0, 0.0, 1.0]   # action 0, from state 2
    ]),
    np.array([
        [0.0, 1.0, 0.0],  # action 1, from state 0
        [0.0, 0.0, 1.0],  # action 1, from state 1
        [0.0, 0.0, 1.0]   # action 1, from state 2
    ])
]

# Reward matrix: each state has a reward per action
R = np.array([
    [5, 10],    # state 0: reward for action 0 and 1
    [-1, 2],    # state 1
    [0, 0]      # state 2 (terminal)
])
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()

print("Policy (best action per state):", vi.policy)
print("Value function:", vi.V)

