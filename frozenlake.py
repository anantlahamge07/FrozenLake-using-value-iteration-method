import numpy as np
import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter


# Our environment's name
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

# Since both our environment and action spaces are of Box classes both our states and actions are represented by integers
State = int
Action = int
# Defining the key for the reward and transition table
RewardKey = tt.Tuple[State, Action, State]
TransitionKey = tt.Tuple[State, Action]


