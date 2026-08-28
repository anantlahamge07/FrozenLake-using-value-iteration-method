from frozenlake import Agent
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym


# Our environment's name
ENV_NAME = "FrozenLake-v1"
# The number of test episodes
TEST_EPISODES = 20

# The overall logic of the whole method is simple. I will explain this in the following different steps
# 1. we will play some random steps(100 in our case) so that we can fill our reward and transition tables with initial values
# 2. After this we will perform a value iteration loop over all the states, updating the value of each state, which means also updating the value table
# 3. The we will play several full episodes to check our improvements using the updated value table. Note we will also be updating our reward and transition tables in this step.
# 4. If the average reward for these test episodes is above 0.8 we will stop training

class Main():


    if __name__ == "__main__":
        # our agent
        agent = Agent()
        # test environment
        test_env = gym.make("FrozenLake-v1")
        # Summary writer for tensorboard for monitoring
        writer = SummaryWriter(comment = "-value-iteration")

        # total average reward
        reward = 0.0
        # iteration number
        iter_no = 0
        best_reward = 0.0
        while True:
            # incrementing the iteration counter
            iter_no += 1
            # now first we have to play 100 random steps so that we can fill our reward and transition tables
            agent.play_n_random_steps(100)
            # now we will run value iteration over all the states
            agent.value_iteration()
            # now we will play test episodes
            for _ in range(TEST_EPISODES):
                # reward for each episode
                episode_reward = 0.0
                # playing the current episode with our test environment
                episode_reward = agent.play_episode(test_env)
                # incrementing the total average reward counter
                reward += episode_reward
                reward /= TEST_EPISODES
                # writing the value to tensorboard
                writer.add_scalar("average reward", reward, iter_no)
                if reward > best_reward:
                    print(f"best reward updated {best_reward} -> {reward}")
                    best_reward = reward
                # if the average reward is greater than 0.8 we will stop training
                if reward > 0.8:
                    print(f"Solved in {iter_no} iterations :)")
                    break
        writer.close()


