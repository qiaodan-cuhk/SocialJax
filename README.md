# SocialJax



*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*



*Common Rewards* : a scenario where all agents share a single, unified reward signal. This approach ensures that all agents are aligned towards achieving the same objective, promoting collaboration and coordination among them.



***Individual Rewards***: each agent is assigned its own reward, inherently encouraging selfish behavior.

SocialJax leverages JAX's high-performance GPU capabilities to accelerate multi-agent reinforcement learning in sequential social dilemmas. We are committed to providing a more efficient and diverse suite of environments for studying social dilemmas. We provide JAX implementations of the following environments: Coins, Commons Harvest: Open, Commons Harvest: Closed, Clean Up, Territory, and Coop Mining, which are derived from [Melting Pot 2.0](https://github.com/google-deepmind/meltingpot/) and feature commonly studied mixed incentives.

Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.

## Update

***[2025/05/28]*** ✨ Updated [SVO](https://github.com/cooperativex/SocialJax/tree/main/algorithms/SVO) algorithm for all environments.

***[2025/04/29]*** 🚀 Updated [Mushrooms](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/mushrooms) environment.

***[2025/04/28]*** 🚀 Updated [Gift Refinement](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/gift) environment.

***[2025/04/16]*** ✨ Added [MAPPO](https://github.com/cooperativex/SocialJax/tree/main/algorithms/MAPPO) algorithm for all environments.

## Installation

First: Clone the repository

Second: Environment Setup.

Option one: Using peotry, make sure you have python 3.10

1. Install Peotry
2. Install requirements     
3. Run code

Option two: conda with requirements.txt

1. Conda
2. Install requirements
3. Run code

Option three: conda with environments.yml

1. Install requirements
2. Run code

## Environments

We introduce the environments and use Schelling diagrams to demonstrate whether the environments are social dilemmas. 


| Environment                  | Description                                                                                       | Schelling Diagrams Proof |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------ |
| Coins                        | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coins)          | &check;                  |
| Commons Harvest: Open        | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest) | &check;                  |
| Commons Harvest: Closed      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest) | &check;                  |
| Commons Harvest: partnership | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest) | &check;                  |
| Clean Up                     | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/cleanup)        | &check;                  |
| Territory                    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/territory)      | &cross;                  |
| Coop Mining                  | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coop_mining)    | &check;                  |
| Mushrooms                    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/mushrooms)      | &check;                  |
| Gift Refinement              | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/gift)           | &check;                  |
| Prisoners Dilemma: Arena     | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/pd_arena)       | &check;                  |


#### Important Notes:

- *Due to algorithmic limitations, agents may not always learn the optimal actions. As a result, Schelling diagrams can prove that the environment is social dilemmas, but they cannot definitively prove that the environment is not social dilemmas.*
- *Territory might not be Social diagram, but as long as the agents' behaviors are interesting, Territory holds intrinsic value.*

## Quick Start

SocialJax interfaces follow [JaxMARL](https://github.com/FLAIROx/JaxMARL/) which takes inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax).

### Make an Environment

You can create an environment using the `make` function:

```python
import jax
import socialjax

env = make('clean_up')
```

### Example

Find more fixed policy [examples](https://github.com/cooperativex/SocialJax/tree/main/fixed_policy).

```python
import jax
import socialjax
from socialjax import make

num_agents = 7
env = make('clean_up', num_agents=num_agents)
rng = jax.random.PRNGKey(259)
rng, _rng = jax.random.split(rng)

for t in range(100):
     rng, *rngs = jax.random.split(rng, num_agents+1)
     actions = [jax.random.choice(
          rngs[a],
          a=env.action_space(0).n,
          p=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
     ) for a in range(num_agents)]

     obs, state, reward, done, info = env.step_env(
          rng, old_state, [a for a in actions]
            )
```

### Speed test

You can test the speed of our environments by running [speed_test_random.py](https://github.com/cooperativex/SocialJax/blob/main/speed_test/speed_test_random.py) or using the [colab](https://colab.research.google.com/github/cooperativex/SocialJax/blob/main/speed_test/speed_test_random.ipynb).

## See Also

[JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.

[PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.