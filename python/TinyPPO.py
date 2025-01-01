# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:57:32 2023

@author: Plutonium
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from trade_evo import TradingEnv
from experience_buffer import Buffer
import time

from torchviz import make_dot
import os
import pandas as pd

import matplotlib.pyplot as plt

num_envs = 1024
horizon = 12
gamma = 0.99

num_epochs = 100
epoch_steps = 1000


entropy_coff_inital = 0.0
clip_range = 0.2
            
torch.set_default_device('cuda')

plt.close('all')

#%%
                          
mse_loss = torch.nn.MSELoss()
data_path = "preprocessed_stock_data.csv"
preprocessed_stock_data_df = pd.read_csv(data_path)

# Set parameters
initial_balance_range = [1000, 10000]  # Initial cash balance
transaction_cost = 0.001  # 0.1% transaction fee
window_size = 1  # Number of past days to include in state

# Initialize the trading environment
env = TradingEnv(data=preprocessed_stock_data_df, 
                 num_envs = num_envs, 
                 initial_balance_range=initial_balance_range, 
                 transaction_cost=transaction_cost, 
                 window_size=window_size,
                 buffer_horizon=10)


replay_buffer = Buffer(env.buffer_hor, env.num_envs, env.num_actions, env.num_states, 0.99)

env.register_buffer(replay_buffer)
print("Initial state:", env.state.shape)



log_probs = torch.zeros((num_envs, horizon))
log_probs_old = torch.zeros((num_envs, horizon)).detach()

#%%
print("State Space : ",env.num_states)
print("Num Ticker Catigories : ",(((env.num_states-2)/window_size)-11)/28)


#%%
# Define model
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = env.num_states
        n_actions = env.num_actions
        layer1_count = 512
        layer2_count = 256
        layer3_count = 128

        
        self.shared1 = nn.Sequential(
                                    nn.Linear(n_features, layer1_count),
                                    nn.ELU()
                                    )
        
        self.shared2 = nn.Sequential(
                                    nn.Linear(layer1_count+n_features, layer2_count),
                                    nn.ELU()
                                    )
        
        self.policy1 = nn.Sequential(
                                    nn.Linear(layer2_count+n_features, layer3_count),
                                    nn.ELU()
                                    )
        self.policy2 = nn.Sequential(
                                    nn.Linear(layer3_count, n_actions),
                                    nn.Tanh(),
                                    )
        
        self.value1 = nn.Sequential(
                                    nn.Linear(layer2_count+n_features, layer3_count),
                                    nn.ELU()
                                    )
        self.value2 = nn.Sequential(
                                    nn.Linear(layer3_count, 1),
                                    )
        
        # self.policy = nn.Sequential(
        #                             nn.Linear(n_features, layer1_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer1_count, layer2_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer2_count, n_actions),
        #                             nn.Tanh(),
        #                             )
        # self.value = nn.Sequential(
        #                             nn.Linear(n_features, layer1_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer1_count, layer2_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer2_count, 1),
        #                             )
        
        # Add a trainable log-std parameter
        self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        s1 = torch.cat((self.shared1(x), x), dim=-1)
        s2 = torch.cat((self.shared2(s1), x), dim=-1)
        # v1 = torch.cat((self.value1(s2) , x), dim=-1)
        v1 = self.value1(s2)
        v2 = self.value2(v1) 
        # p1 = torch.cat((self.policy1(s2), x), dim=-1)
        p1 = self.policy1(s2)
        p2 = self.policy2(p1)        
        # p = self.policy(x)
        # v = self.value(x)
        return v2, p2, self.log_std
    

    
#%%
# a = torch.rand([3,5])
Agent = Policy()
agent_optim = optim.Adam(Agent.parameters(), lr=3e-4)

# # actor_optim = optim.Adam( [ { 'params': PolicyNet1.actor_top.parameters() },
# #                             { 'params': PolicyNet1.actor_mean.parameters() },
# #                             { 'params': PolicyNet1.actor_std.parameters()} ], 
# #                             lr=1e-3)
          

# Agent.load_state_dict(torch.load("model_name.pth"))



#%%
import os
from torch.utils.tensorboard import SummaryWriter

def create_unique_log_dir(base_name="runs/ppo_training"):
    # Start with the base directory name
    log_dir = base_name
    counter = 1
    
    # Increment the directory name if it already exists
    while os.path.exists(log_dir):
        log_dir = f"{base_name}_{counter}"
        counter += 1
    
    return log_dir
log_dir = create_unique_log_dir()
writer = SummaryWriter(log_dir=log_dir)
print(f"Logging to: {log_dir}")

average_rewards_per_epoch = []
average_episode_lengths = []

# Reset Environments
env_ids = torch.arange(env.num_envs)
env.reset_idx(env_ids)


for epoch in range(num_epochs):

    # PolicyNet2.load_state_dict(PolicyNet1.state_dict())
    # Critic2.load_state_dict(Critic.state_dict())
    actor_loss_list = []
    critic_loss_list = []
    episode_rewards = []
    episode_lengths = []
    

    for _ in range(epoch_steps):
    # for _ in range(minibatch_steps):
        # s1, a1, r1, s2, d2, log_probs_old, returns = env.buffer.get_SARS_minibatch(minibatch_size) 
        s1, a1, r1, s2, d2, log_probs_old, returns = replay_buffer.get_SARS()

        
        [vals_s1, probs_s1] = Agent(s1)
        
        
        # action_pd_s1 = torch.distributions.Normal(probs_s1[:,:,0], probs_s1[:,:,1])
        
        action_pd_s1 = torch.distributions.Normal(probs_s1, 0.025*torch.ones_like(probs_s1.detach()))
        

        # td_error = returns - vals_s1.squeeze(-1)
        advantage = returns - vals_s1.squeeze(-1)
        

        # normalize advantage... (Doesn't Seem to work)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        log_probs = action_pd_s1.log_prob(a1)
        ratio = torch.exp(log_probs - log_probs_old)
        

        entropy_coff = entropy_coff_inital * (1-(epoch/num_epochs))
        # entropy_loss = -action_pd_s1.entropy().mean() * entropy_coff

        # entropy_loss = -entropy_coff*torch.mean(-log_probs)
        entropy_loss = -entropy_coff * action_pd_s1.entropy().mean()

        policy_loss_1 = advantage.view(env.num_envs, env.buffer_hor, 1).detach() * ratio
        policy_loss_2 = advantage.view(env.num_envs, env.buffer_hor, 1).detach() * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() 
        
        value_loss = mse_loss(vals_s1.squeeze(-1), returns)
        
        total_loss = policy_loss + value_loss*0.2 + entropy_loss
        
        agent_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.parameters(), 0.5) #Max Grad Norm
        agent_optim.step()

        
        actor_loss_list.append(policy_loss.detach().cpu().numpy())
        critic_loss_list.append(value_loss.detach().cpu().numpy())
        



        
        
        # NAN Checker
        for name, param in Agent.named_parameters():
            if( torch.any(torch.isnan(param)) ):
                print(name)
                print(param)
                raise ValueError(f"NaN detected in parameter: {name}")

    
        # Advance Environments  1 Step
        with torch.no_grad():        
            s1, a1, r1, s2, d2, log_probs_old, returns = replay_buffer.get_SARS()
            [vals, probs] = Agent(s2)
            newest_probs = probs[:,0,:]
            action_pd = torch.distributions.Normal(newest_probs, 0.01*torch.ones_like(newest_probs))
            next_actions = action_pd.sample()
            log_probs_sample = action_pd.log_prob(next_actions)
            env.step(next_actions, log_probs_sample, Agent)
       
            # print(env.done)
            # print(env.elapsed_days)
            # print(next_actions)
            
            
            env_ids = env.done.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                episode_lengths.extend(env.elapsed_days[env_ids].cpu().numpy().tolist())
                episode_rewards.extend(env.reward[env_ids].cpu().numpy().tolist())
                env.reset_idx(env_ids)
        
        
    # Compute average episode length
    average_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    average_episode_lengths.append(average_episode_length)
    
    # Log average reward for the epoch
    average_rewards_per_epoch.append(np.mean(episode_rewards))
    
    # Logging epoch details to the console
    print(f"Epoch {epoch}")
    print(f"Policy Loss Avg: {np.mean(actor_loss_list)}. Value Loss Avg: {np.mean(critic_loss_list)}. Avg Returns: {returns.mean()}")
    print(f"Avg Rewards for Epoch: {average_rewards_per_epoch[-1]}")
    print(f"Average Episode Length: {average_episode_length}")
    
    # Logging to Tensorboard
    writer.add_scalar("Loss/Policy", np.mean(actor_loss_list), epoch)
    writer.add_scalar("Loss/Value", np.mean(critic_loss_list), epoch)
    writer.add_scalar("Reward/Average per Epoch", average_rewards_per_epoch[-1], epoch)    
    writer.add_scalar("Episode/Average Length", average_episode_length, epoch)
    
    with torch.no_grad():
        # Useful extra info
        approx_kl1 = ((torch.exp(ratio) - 1) - ratio).mean() #Stable Baselines 3
        approx_kl2 = (log_probs_old - log_probs).mean()    #Open AI Spinup
        # print('kl approx : {} : {} : {}'.formaWDt(approx_kl1, approx_kl2, ratio.mean()))
    
    # print(Qvals.reshape([siz,siz]))
    
    
#%%
torch.save(Agent.state_dict(), "model_name.pth")

#%%

# Reset Environments
env_ids = torch.arange(env.num_envs)
env.reset_idx(env_ids)

done = False
total_reward = 0
portfolio_values = []
actions_taken = []

# Run one episode
env.max_trading_days = 1000

state = env.state
[vals_s1, probs_s1] = Agent(state)

# while not done:
#     state = env.state
#     [vals_s1, probs_s1] = Agent(state)
    
    # env.step(next_actions, log_probs_sample, Agent)

    # # print(env.done)
    # # print(env.elapsed_days)
    # # print(next_actions)
    
    
    # env_ids = env.done.view(-1).nonzero(as_tuple=False).squeeze(-1)
    # if len(env_ids) > 0:
    #     episode_lengths.extend(env.elapsed_days[env_ids].cpu().numpy().tolist())
    #     env.reset_idx(env_ids)
        
    # action, _states = model.predict(obs, deterministic=True)
    # obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated or truncated

    # total_reward += reward
    # portfolio_value = env.balance + env.shares * env._get_price()
    # portfolio_values.append(portfolio_value)
    # actions_taken.append(action)  # Record actions

    # env.render()  # Optional: Display the current state

#%%
# Convert actions to a NumPy array for easier plotting
actions_taken = np.array(actions_taken)

# Create the subplots
fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot portfolio value over time
ax[0].plot(portfolio_values, label="Portfolio Value", color="blue")
ax[0].set_title("Portfolio Value Over Time")
ax[0].set_ylabel("Portfolio Value")
ax[0].legend()

# Plot actions over time
ax[1].plot(actions_taken[:, 0], label="Sell Action (Fraction of Shares)", color="orange")
ax[1].plot(actions_taken[:, 1], label="Buy Action (Fraction of Balance)", color="green")
ax[1].set_title("Actions Over Time")
ax[1].set_xlabel("Step")
ax[1].set_ylabel("Action Value")
ax[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#%% Environment Testing
env_ids = torch.arange(env.num_envs)
env.reset_idx(env_ids)

print("Observation shape:", env.state.shape)
print("Observation dtype:", env.state.dtype)
print("Observation:", env.state)

print("Initial balance:", env.balance)
print("Initial shares:", env.shares)

for action in [[0.0, 0.0], [0.5, 0.5], [0.5, 1.0]]:
    print("-"*20)
    print("Initial balance:", env.balance)
    print("Initial shares:", env.shares)
    print(f"Action: {action}")
    
    action_tensor = torch.tensor([action] * env.num_envs, dtype=torch.float32)
    print("Action Tensor:", action_tensor)
    
    print("Sell Amount:", action[0] * env.shares)
    print("Buy Amount:", action[1] * env.balance / env._get_price())
    print("Current Price:", env._get_price())
    print("")
    
    with torch.no_grad():
        log_probs_sample = torch.zeros_like(action_tensor)  # Placeholder for log_probs, no actual sampling here
        env.step(action_tensor, log_probs_sample, Agent)
            
    print("New balance:", env.balance)
    print("New shares:", env.shares)
    
    print(f"Reward: {env.reward}, Done: {env.done}")
    print("")    
    portfolio_value = env.balance + env.shares * env._get_price()
    print(f"Portfolio Value: {portfolio_value}")

    


