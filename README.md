### Imitation Learning, RL, Policy Gradients
This repository contains implementation of Imitation Learning, Policy Gradients, Actor Critic and Soft Actor Critic Methods. Exeriments have been performed on 2 Mujoco environments namely Hopper-v4 and Ant v-4 environments.

## Imitation Learning
- The imitation agent is a simple MLP used to learn the policy from the expert with Dagger.
- With probability of beta, use the expert policy and with 1-beta use action from the policy being trained.
- Sample 400 trajectories and train your policy for 200 iterations. 
- Avg reward of 2500 in Hopper and 1500 in Ant can be obtained

## Policy Gradient and Soft Actor Critic
- Here I first tried the Vannila Policy Gradient and Actor Critic Method for the Hopper-v4 environment
- These methods arent able to learn the dynamics of the environment
- I then have implemented the Soft Actor Critic Method to maximize the avg reward
- Details of the Soft Actor Critic is here [click here](https://arxiv.org/abs/1801.01290)

## Stable Baselines
- Here the popular stable baselines library has been used to experiment with popular RL techniques on Hopper, Ant and Panda Push Environments
- I have experimented with PPO and Soft Actor Critic Implementations available in stable baselines 

## How to Run
- First create a conda environment and then install the packages
```bash
conda env create -f rl_env_lin.yml
conda activate  rl_env
pip install -e .
```
- For training the agent, run the following command,
```bash
	python scripts/train_agent.py --env_name <ENV> --exp_name <ALGO>  [optional tags]
```
## Arguments
- **env_name**: Hopper-V4, Ant-V4
- **exp_name**: either imitation, RL, or stable baselines
- **optional_tags**: train with GPU and logs


