# FrozenLake Using Value Iteration

This project solves Gymnasium's `FrozenLake-v1` environment using a simple
model-based value iteration approach.

The agent first explores the environment with random actions to estimate:

- the reward for each `(state, action, next_state)` transition
- how often each transition happens
- the value of each state

After collecting transition data, it repeatedly updates the state-value table
and evaluates the learned greedy policy.

## Project Structure

```text
.
├── frozenlake.py   # Agent implementation and value iteration logic
├── main.py         # Training loop and TensorBoard logging
├── LICENSE        # MIT license
└── README.md
```

## How It Works

The main training loop follows these steps:

1. Create an `Agent` for the `FrozenLake-v1` environment.
2. Play random steps to fill the reward and transition tables.
3. Run value iteration across all states.
4. Evaluate the current greedy policy over several test episodes.
5. Log the average reward to TensorBoard.
6. Stop when the average reward is greater than `0.8`.

The value update uses:

```text
Q(s, a) = sum P(s' | s, a) * (R(s, a, s') + gamma * V(s'))
V(s) = max_a Q(s, a)
```

where `gamma` is the discount factor.

## Requirements

- Python 3.10+
- Gymnasium
- NumPy
- PyTorch, for TensorBoard's `SummaryWriter`

Install the dependencies with:

```bash
pip install gymnasium numpy torch tensorboard
```

## Running the Project

Start training with:

```bash
python main.py
```

During training, the program prints whenever it reaches a new best average
reward:

```text
1. best reward updated 0.0 -> 0.05
...
Solved in 42 iterations :)
```

The exact number of iterations can change from run to run because FrozenLake is
stochastic and the transition table is estimated from sampled experience.

## TensorBoard

Training logs are written to the `runs/` directory. To view them, run:

```bash
tensorboard --logdir runs
```

Then open the URL printed by TensorBoard in your browser.

## Main Files

### `frozenlake.py`

Contains the `Agent` class:

- `play_n_random_steps(n)`: collects random experience
- `calc_action_value(state, action)`: estimates the Q-value for an action
- `best_action_to_take(state)`: chooses the action with the highest estimated value
- `play_episode(env)`: evaluates the greedy policy for one episode
- `value_iteration()`: updates the value table for every state

### `main.py`

Runs the full training loop:

- creates the training and test environments
- collects random samples
- applies value iteration
- evaluates the policy
- logs average reward to TensorBoard

## Notes

This implementation estimates the transition model from experience instead of
directly reading Gymnasium's internal transition probabilities. That makes it a
good learning example for model-based reinforcement learning, but performance
depends on how much random exploration data has been collected.

The default constants are:

```python
GAMMA = 0.9
TEST_EPISODES = 20
ENV_NAME = "FrozenLake-v1"
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
