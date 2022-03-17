# Super Auto Pets Gym

`sapai-gym` provides an [OpenAI Gym](https://github.com/openai/gym) environment for [Super Auto Pets](https://teamwoodgames.com/).
This Gym environments provide a standard interface to train reinforcement learning (RL) models for Super Auto Pets and
is compatible with any tools that accept Gym interfaces, including [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

`sapai-gym` is built on top of [sapai](https://github.com/manny405/sapai), which is a python implementation of
Super Auto Pets.

## Installation

Clone the repo and install dependencies

```shell
git clone https://github.com/alexdriedger/sapai-gym.git
cd sapai-gym
python setup.py install
```

## Example Usage

`SuperAutoPetsEnv` implements the `gym.Env` interface. Here is a basic example that takes random actions

```python
from sapai_gym import SuperAutoPetsEnv

def opponent_generator(num_turns):
    # Returns teams to fight against in the gym 
    return []

env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=False)
obs = env.reset()

for step_num in range(1000):
    if step_num % 100 == 0:
        print(f"Step {step_num}")

    # Random actions
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
env.close()
```

## Training RL Agent Using Stable Baselines3

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

def train_with_masks():
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
    obs = env.reset()
    
    num_games = 0
    while num_games < 100:
        # Predict outcome with model
        action_masks = get_action_masks(t_env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, done, info = env.step(action)
        if done:
            num_games += 1
            obs = env.reset()
    env.close()
```

## Action & Observation Space

The action space is a `Discrete` space of the total number of possible actions in Super Auto Pets (63 different actions
in total without counting freezing and rearranging teams). For example, there are at most 6 pets available in the shop
for purchase, so there are 6 `buy_pet` actions in the action space.

For observations, categorical features (pet names, pet statuses, and food names) are one-hot encoded. Attack and health
are divided by 50, to remain in [0, 1]. All other features are scaled to [0, 1].

## Opponent Generation

If Super Auto Pets, when you end your turn in the shop, you fight an opponent. The question of how to generate this
opponent when simulating Super Auto Pets in a controlled environment is interesting and could have multiple different answers and implementations.

In order to allow flexibility in `sapai-gym`, an opponent generator is passed into the environment, which the environment
uses to generate the opponents the agent will play. The simplest form of an opponent generator simply returns a static set of teams,
which are the same every game. More complicated opponent generators could generate teams like a smart opponent would.

