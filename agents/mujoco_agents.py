import itertools
import torch
import random
import torch.nn as nn
import torch
import numpy as np
from torch import optim
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal
from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3 import PPO, SAC, HerReplayBuffer, TD3


class Imitiation_learner(nn.Module):
    def __init__(self, observation_dim:int, action_dim:int, hidden_dim:int):
        super(Imitiation_learner, self).__init__()
        self.layer1 = nn.Linear(observation_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, action_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x = F.relu(self.ln(self.layer1(x)))
        x = F.relu(self.ln1(self.layer2(x)))
        x = F.relu(self.ln2(self.layer3(x)))
        x = torch.tanh(self.layer4(x))

        return x



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.hidden_dim = self.hyperparameters['hidden_size']
        self.max_length = self.hyperparameters['max length']
        self.num_trajectories = self.hyperparameters['num trajectories']
        self.replay_buffer = ReplayBuffer(max_size=1000000) #you can set the max size of replay buffer if you want
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imitation_learner = Imitiation_learner(observation_dim, action_dim, self.hidden_dim)
        self.optimizer = optim.AdamW(self.imitation_learner.parameters(), lr=1e-4, amsgrad=True)
        self.mse_loss = nn.MSELoss()
        self.beta = None
        self.total_reward = 0
        self.batch_size = self.hyperparameters['batch_size']
        self.imitation_learner = self.imitation_learner.to(self.device)

    def forward(self, observation: torch.FloatTensor):
        return self.imitation_learner(observation)
    
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        
        return self.imitation_learner(observation)
    
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        print('training the agent')
        if not hasattr(self, "expert_policy"):
            print('loading expert policy')
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            print('adding rollouts')
            self.replay_buffer.add_rollouts(initial_expert_data)
        print('starting to train')
        self.beta = 1/(1+ envsteps_so_far/1000)
        trajectories =  utils.sample_n_trajectories(env, self, self.num_trajectories, self.max_length, False)
        train_loss = 0
        for i in tqdm(range(len(trajectories))):
            for j in tqdm(range(len(trajectories[i]['action']))):
                observation = torch.from_numpy(trajectories[i]['observation'][j])
                observation = observation.to(self.device)
                if random.random()<self.beta:
                    trajectories_new = torch.from_numpy(trajectories[i]['observation'][j])
                    trajectories_new = trajectories_new.to(self.device)
                    action = self.expert_policy.get_action(trajectories_new)
                    action = torch.tensor(action)
                    action = action.to(self.device)
                else:
                    action = trajectories[i]['action'][j]
                    action = torch.tensor(action)
                    action = action.to(self.device)
                y_pred = self.imitation_learner(observation)
                y_pred = y_pred.to(self.device)
                loss = self.mse_loss(y_pred, action)
                train_loss+=loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        replay_trajectories = self.replay_buffer.sample_batch(self.batch_size)

        for i in tqdm(range(len(replay_trajectories['acs']))):
            observation = torch.from_numpy(replay_trajectories['obs'][i])
            observation = observation.to(self.device)
            if random.random()<self.beta:
                replay_trajectories_new = torch.from_numpy(replay_trajectories['obs'][i])
                replay_trajectories_new = replay_trajectories_new.to(self.device)
                action = self.expert_policy.get_action(replay_trajectories_new)
                action = torch.tensor(action)
                action = action.to(self.device)
            else:
                action = replay_trajectories['acs'][i]
                action = torch.tensor(action)
                action = action.to(self.device)
            
            y_pred = self.imitation_learner(observation)
            y_pred = y_pred.to(self.device)
            loss = self.mse_loss(y_pred, action)
            train_loss+=loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        eval_reward = np.mean(eval_returns)
        if eval_reward>self.total_reward:
            self.total_reward = eval_reward
            save_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            best_model_path = os.path.join(save_dir_path, f"Imitation_{self.args.env_name}.pth")
            torch.save(self.imitation_learner.state_dict(), best_model_path)
            print('model saved')
        
        return {'episode_loss': train_loss, 'eval_reward': eval_reward, 'trajectories': trajectories, 'current_train_envsteps': self.num_trajectories} #you can return more metadata if you want to




#TODO IMPLEMENT SOFT ACTOR CRITIC

class SACActor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(SACActor, self).__init__()
        # self.observation_dim = observation_dim
        # self.action_dim = action_dim
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.means = nn.Linear(256, action_dim)
        self.variances = nn.Linear(256, action_dim)
        self.relu = nn.LeakyReLU()
    def forward(self,observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        means = self.means(x)
        variance = self.variances(x)
        variance = torch.clamp(variance, min = 0.0001, max = 1)
        return means, variance

class SACCritic(nn.Module):
    def __init__(self,observation_dim, action_dim):
        super(SACCritic, self).__init__()
        self.fc1 = nn.Linear(observation_dim+action_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.final_layer = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, states, actions):
        x = self.relu(self.fc1(torch.cat([states, actions], dim=2)))
        x = self.relu(self.fc2(x))
        x = self.final_layer(x)

        return x
    

class SACValue(nn.Module):
    def __init__(self, observation_dim):
        super(SACValue, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256,1)
        self.relu = nn.LeakyReLU()

    def forward(self, observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        value = self.value(x)
        return value









class RLAgent(BaseAgent):

    '''
    Please implement an policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.actor = SACActor(observation_dim, action_dim)
        self.critic_1 = SACCritic(observation_dim, action_dim)
        self.critic_2 = SACCritic(observation_dim, action_dim)
        self.value = SACValue(observation_dim)
        self.target_value = SACValue(observation_dim)
        self.optimizer_critic_1 = optim.AdamW(self.critic_1.parameters(), lr = 0.0003)
        self.optimizer_critic_2 = optim.AdamW(self.critic_2.parameters(), 0.0003)
        self.optimizer_actor = optim.AdamW(self.actor.parameters(), lr = 0.0003)
        self.optimizer_value = optim.AdamW(self.value.parameters(), lr = 0.0003)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_scale = 2
        self.gamma = 0.99
        self.total_reward = 0
        self.actor = self.actor.to(self.device)
        self.critic_1 = self.critic_1.to(self.device)
        self.critic_2 = self.critic_2.to(self.device)
        self.value = self.value.to(self.device)
        self.target_value = self.target_value.to(self.device)
        self.tau = 0.005
        self.update_target(tau=1)

    

    def forward(self, observation):
        means, vars = self.actor(observation)
        action_distributions = Normal(means, vars)
        action = action_distributions.rsample()
        actions = torch.tanh(action).to(self.device)
        return actions
    
    def update_action(self, observation):
        means, vars = self.actor(observation)
        action_distributions = Normal(means, vars)
        action = action_distributions.rsample()
        action_new = action.squeeze(0)
        actions = torch.tanh(action).to(self.device)
        actions_new = actions.squeeze(0)
        log_probs = action_distributions.log_prob(action_new) - torch.log( 1- actions_new.pow(2) + 0.0001) 
        log_probs = log_probs.squeeze(0)
        log_probs = log_probs.sum(1, keepdim = True)
        return actions, log_probs
    
    
    def update_target(self, tau=None):
        if tau is None:
            tau = self.tau  # Ensure tau is set; fallback to instance attribute self.tau
        current_state_dict = self.value.state_dict()
        target_state_dict = self.target_value.state_dict()

        # Perform the weighted update of parameters
        for key in target_state_dict:
            updated_weights = tau * current_state_dict[key].clone() + (1 - tau) * target_state_dict[key].clone()
            target_state_dict[key] = updated_weights

        # Load the updated state dictionary into the target model
        self.target_value.load_state_dict(target_state_dict)
    
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        means, vars = self.actor(observation)
        action_distributions = Normal(means, vars)
        action = action_distributions.sample()
        actions = torch.tanh(action).to(self.device)
        return actions
    



    
    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #TODO WRITE TRAINING LOOP FOR SOFT ACTOR CRITIC
        trajectories = utils.sample_n_trajectories(env, self, 50, 3000, False)
        loss = 0
        for trajectory in trajectories:
            print('SAC Training')
            observation = trajectory['observation']
            observation_tensor = torch.tensor(observation)
            observation_tensor = observation_tensor.unsqueeze(0)
            observation_tensor = observation_tensor.to(self.device)
            reward = torch.from_numpy(trajectory['reward'])
            reward = reward.unsqueeze(0)
            reward = reward.to(self.device)
            next_state = torch.from_numpy(trajectory['next_observation'])
            next_state = next_state.unsqueeze(0)
            next_state = next_state.to(self.device)
            action = torch.from_numpy(trajectory['action'])
            action = action.unsqueeze(0)
            action = action.to(self.device)
            value_current = self.value(observation_tensor).view(-1)  
            value_next = self.target_value(next_state).view(-1)
            value_next[-1]=0    
            actions, log_probs = self.update_action(observation_tensor)
            q1_np = self.critic_1.forward(observation_tensor,actions)
            q2_np = self.critic_2.forward(observation_tensor,actions)
            critic_value = torch.min(q1_np, q2_np) 
            critic_value = critic_value.view(-1)
            self.optimizer_value.zero_grad()
            value_target = critic_value - log_probs.view(-1)
            value_loss = 0.5 * F.mse_loss(value_current, value_target)
            value_loss.backward(retain_graph=True)
            self.optimizer_value.step()
            actor_loss = torch.mean(log_probs- critic_value)
            loss+=actor_loss.item()
            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer_actor.step()
            self.optimizer_critic_1.zero_grad()
            self.optimizer_critic_2.zero_grad()
            q_hat = self.reward_scale * reward + self.gamma * value_next
            q_hat = q_hat.view(-1)
            q1_op = self.critic_1.forward(observation_tensor,action).view(-1)
            q2_op = self.critic_2.forward(observation_tensor,action).view(-1)
            critic_loss = 0.5 * (F.mse_loss(q1_op, q_hat) + F.mse_loss(q2_op, q_hat))
            critic_loss.backward()
            self.optimizer_critic_1.step()
            self.optimizer_critic_2.step()

            self.update_target()


        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = utils.sample_trajectories(env, self.get_action, 15*max_ep_len, max_ep_len)
        eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]
        current_reward = np.mean(eval_returns)
  
        print(current_reward)
        if current_reward>self.total_reward:
            self.total_reward = current_reward
            save_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            best_model_path = os.path.join(save_dir_path, f"RL_{self.args.env_name}.pth")
            torch.save(self.imitation_learner.state_dict(), best_model_path)
            print('model saved')

            
        return  {'episode_loss': loss, 'trajectories': trajectories, 'current_train_envsteps':50}


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
class SaveModelCallback(BaseCallback):
    def __init__(self,eval_env,env_name, eval_freq, save_path=None):
        super(SaveModelCallback, self).__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.env_name = env_name
        if self.env_name == 'PandaPush-v3':
            self.max_reward =  -100
        else:
            self.max_reward = 0

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps != 0 and self.eval_freq != 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env)
            print(f"Num timesteps: {self.num_timesteps}, Mean reward: {mean_reward}")

            if self.save_path is not None and self.max_reward<mean_reward:
                print(f"Saving model at {self.save_path}")
                self.max_reward = mean_reward
                print(self.max_reward)
                self.model.save(self.save_path)

        return True  # Continue training




class SBAgent(BaseAgent):
    def __init__(self, env_name, **hyperparameters):
        #implement your init function
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
        from stable_baselines3.common.noise import NormalActionNoise
        import panda_gym
        import gymnasium 
        import os
        self.env_name = env_name
        self.env = make_vec_env(env_name,n_envs = 1)
        if self.env_name == 'Ant-v4':
            self.model =  PPO(policy="MlpPolicy",batch_size= 2048,gae_lambda= 0.95,learning_rate= 0.0003,max_grad_norm=0.5,env= self.env, verbose=1, tensorboard_log = 'tensorboard_logs')
            save_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
            best_model_path = os.path.join(save_dir_path, f"sb_{self.env_name}.pth")
            self.save_callback = SaveModelCallback(self.env,self.env_name, eval_freq=5000, save_path=best_model_path)
        
        else:
           n_actions = self.env.action_space.shape[-1]
           action_noise = NormalActionNoise(mean=0, sigma=0.1 * np.ones(n_actions))
           goal_selection_strategy = "future"
           save_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
           best_model_path = os.path.join(save_dir_path, f"sb_{self.env_name}.pth")
           self.model =   SAC("MultiInputPolicy",batch_size=  1024,buffer_size= 1000000,gamma= 0.99,tau= 0.05,learning_rate = 5e-4,train_freq = 1,replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=dict(n_sampled_goal=4,goal_selection_strategy=goal_selection_strategy),learning_starts = 5000,action_noise=action_noise,env= self.env, verbose=1,gradient_steps  = 1,optimize_memory_usage= False, tensorboard_log = 'tensorboard_logs')
           self.save_callback = SaveModelCallback(self.env,self.env_name, eval_freq=1000, save_path=best_model_path)
 
    
    def learn(self):
        if self.env_name == 'Ant-v4':
            self.model.learn(total_timesteps=6000000, progress_bar=True,callback = self.save_callback, log_interval = 5, tb_log_name = f'{self.env_name}-PPO-sb3')
        else:
            self.model.learn(total_timesteps=5e5, progress_bar=True, callback = self.save_callback, log_interval = 5,  tb_log_name = f'{self.env_name}-TD3-sb3')

    
    
    def load_checkpoint(self, checkpoint_path):
        #implement your load checkpoint function
        self.model.policy.load_state_dict(torch.load(checkpoint_path))
    
    
    def get_action(self, observation):
        #implement your get action function
        return self.model.predict(observation)
    