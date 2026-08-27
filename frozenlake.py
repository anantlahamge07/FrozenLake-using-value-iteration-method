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


class Agent():
    def __init__(self):
        # creating the environment
        self.env = gym.make(ENV_NAME)
        # getting the first observation
        self.obs, _ = self.env.reset()
        # The reward table
        self.reward_table = tt.Dict[RewardKey, float]
        # The transition table
        self.transition_table = tt.Dict[TransitionKey, tt.Dict[State, int]]