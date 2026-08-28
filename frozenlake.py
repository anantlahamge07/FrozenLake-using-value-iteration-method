import numpy as np
import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter


# Our environment's name
ENV_NAME = "FrozenLake-v1"
# The discount factor
GAMMA = 0.9
# The number of test episodes
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
        self.reward_table: tt.Dict[RewardKey, float] = defaultdict(float)
        # The transition table
        self.transition_table: tt.Dict[TransitionKey, Counter] = defaultdict(Counter)
        # The value table
        self.value_table: tt.Dict[State, float] = defaultdict(float)


    # This method will be used to play n random steps, so that we can update our reward and transition tables with values
    def play_n_random_steps(self, n: int):
        for _ in range(n):
            # getting the action
            action = self.env.action_space.sample()
            # getting the next observation, action and some flags from the environment
            next_obs, reward, is_done, is_trunc, _ = self.env.step()
            # updating the reward table
            self.reward_table[(self.obs, action, next_obs)] = float(reward)
            # updating the transition table
            self.transition_table[(self.obs, action)][next_obs] +=  1
            # checking whether the episode is done or truncated
            if is_done or is_trunc:
                self.obs, _ = self.env.reset()
            else:
                # if not done updating the current state
                self.obs = next_obs

    # This method will calculate the actual action value a given state action pair (s, a)
    def calc_action_value(self, state: State, action: Action) -> float:
        # getting the value dict from our transition table
        dict = self.transition_table[(state, action)]
        # The number of times the given action got executed
        total = sum(dict.values)
        # The action value
        q = 0.0
        for k, v in self.transition_table.items():
            # getting the reward
            reward = self.reward_table[(self.obs, action, k)]
            # value of the destination state
            value = self.value_table[k]
            # Probability of visitng this destination state
            p = v/total
            # updating the action value
            q += p * (reward + GAMMA * value)
        return q

    # This function decides which is the best action to take, using the maximum action value
    def best_action_to_take(self, state: State) -> Action:
        best_action = None
        max_action_value = None
        for action in range(self.env.action_space.n):
            # getting the action value of the current state, action pair
            action_value = self.calc_action_value(state, action)
            if max_action_value is None or (action_value > max_action_value):
                # Updating the maximum action value and the best action variables
                best_action = action
                max_action_value = action_value
        return best_action


    # This method plays one full episode
    # Note: this method will get a new environment instead of our main environment as an argument, because we don't want to
    # mess with our main environment while playing test episodes, to collect random data
    def play_episode(self, env: gym.Env) -> float:
        # Total reward for the whole episode
        total_reward = 0.0
        # getting the observation from our test environment
        state, _  = env.reset()
        while True:
            # getting the action
            action = env.action_space.sample()
            # getting the next observation, action and some flags from the environment
            next_obs, reward, is_done, is_trunc, _ = env.step()
            # updating the reward table
            self.reward_table[(state, action, next_obs)] = float(reward)
            # updating the transition table
            self.transition_table[(state, action)][next_obs] +=  1
            # updating the total reward
            total_reward += reward
            # checking whether the episode is done or truncated
            if is_done or is_trunc:
                state, _ = self.env.reset()
            else:
                # if not done updating the current state
                state = next_obs
        return total_reward

    # This is the last method of our agent's class
    # This method is surprisingly simple it just performs the value iteration step over all the states of our environment
    # i.e. is updating the value of each state with the maximum action value 
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            # now we just calculate all the action values for all the states reachable from this state
            action_values = [self.calc_action_value(state, action) for action in self.env.action_space.n]
            # updating the state value table for the current state with the maximum action value
            self.value_table[state] = max(action_values)
    