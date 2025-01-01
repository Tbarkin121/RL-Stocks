# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:28:48 2023

@author: Plutonium
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np 
import time
from sklearn.preprocessing import StandardScaler
import pickle
torch.set_default_device("cuda")



#%%

def RandTensorRange(size, min_val, max_val, dtype=torch.float):
    tensor_range = max_val-min_val
    rt = torch.rand(size)*tensor_range + min_val
    rt = rt.to(dtype=dtype)
    return rt

# rez = RandTensorRange( (2,1), 0, 10, dtype=torch.int)
# print(rez)
# rez = RandTensorRange( (2,1), 0, 10).floor()
# print(rez)

#%%            
            
class TradingEnv:
    def __init__(self, data, num_envs=2, initial_balance_range=[0, 10000], transaction_cost=0.001, window_size=10, buffer_horizon=10):
        """
        Initialize the trading environment.
        
        Parameters:
        - data: Preprocessed historical data (pandas DataFrame).
        - num_envs: Number of parallel environments to simulate.
        - initial_balance_range: Range of starting cash balance [min, max].
        - transaction_cost: Proportional transaction fee for buy/sell actions.
        - window_size: Number of past days used for state representation.
        """
        self.data = data
        self.column_indices = {name: idx for idx, name in enumerate(self.data.columns)}
        # Fit scaler on historical stock data (assumes stock data is in numeric columns only)
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        # Save the scaler for future use
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        
        self.data_tensor = torch.tensor(self.data.values, dtype=torch.float32)
        self.data_tensor_scaled = torch.tensor(self.scaled_data, dtype=torch.float32)
        self.num_envs = num_envs
        self.initial_balance_range = initial_balance_range
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.buffer_hor = buffer_horizon
    
        self.min_balance = initial_balance_range[0]

        # Action and state space
        self.num_actions = 2  # [Sell %, Buy %]
        self.num_states = self.data_tensor.shape[1] * self.window_size + 2  # preprocessed ticker data + balance + shares
        
        ticker_data_size = self.data_tensor.shape[1] * self.window_size
        
        # State placeholder
        self.state = torch.zeros((self.num_envs, self.num_states))
        self.ticker_data = self.state[:,0:ticker_data_size].view(-1,ticker_data_size)
        self.balance_scaled = self.state[:,ticker_data_size].view(-1,1)
        self.shares_scaled = self.state[:,ticker_data_size+1].view(-1,1)
        
        self.balance_scale = 1e6
        self.shares_scale = 1e6
        self.reward_scale = 1e6
        
        self.balance = torch.zeros( (self.num_envs,  1))
        self.shares = torch.zeros( (self.num_envs, 1))
        

        
        self.max_trading_days = 365 # Lets say 1 year of trading
        self.num_trading_days = self.data_tensor.shape[0]
        self.starting_day = torch.zeros( (self.num_envs, 1), dtype=torch.int )
        self.elapsed_days = torch.zeros( (self.num_envs, 1), dtype=torch.int )
        env_ids = torch.arange(num_envs)
        self.reset_idx(env_ids)
        
        print('Env Init Complete')
        

    def register_buffer(self, replay_buf):
        self.buffer=replay_buf
        

    def reset_idx(self, env_ids):
        """
        Reset the environment for a new episode.
        """
        self.balance[env_ids, :] = RandTensorRange((len(env_ids), 1), self.initial_balance_range[0], self.initial_balance_range[1])
        self.shares[env_ids, :] = torch.zeros( (len(env_ids), 1))
        self.balance_scaled = self.balance/self.balance_scale
        self.shares_scaled = self.shares/self.shares_scale
        
        self.starting_day[env_ids, :] = RandTensorRange( (len(env_ids), 1), self.window_size, self.num_trading_days-self.max_trading_days, dtype=torch.int)
        self.elapsed_days[env_ids, :] = torch.zeros( (len(env_ids), 1), dtype=torch.int )
        
        # Update ticker_data state for the new day
        current_day = self.starting_day + self.elapsed_days
        for i in range(self.window_size):
            self.ticker_data[:, i*self.data_tensor_scaled.shape[1]:(i+1)*self.data_tensor_scaled.shape[1]] = self.data_tensor_scaled[ current_day.squeeze()+i-self.window_size, :]
                
    def register_buffer(self, replay_buf):
        self.buffer=replay_buf
        
    def step(self, actions, log_probs, ValueNet):
        """
        Perform an action and update the environment state.
        
        Parameters:
        - actions: Tensor of shape [num_envs, 2], where:
            - actions[:, 0]: Sell percentage (0.0 to 1.0)
            - actions[:, 1]: Buy percentage (0.0 to 1.0)
        """
        with torch.no_grad():
            actions = torch.clip(actions, 0.0, 1.0)
            buf_states1 = self.state
            self.buffer.update1(buf_states1, actions, log_probs)
            
            
            current_day = self.starting_day + self.elapsed_days
            close_idx = self.column_indices['AAPL_Close']
            close_data = self.data_tensor[:, close_idx]

                
            current_prices = close_data[current_day]
        
            # Calculate portfolio changes
            # Number of shares to sell
            sell_amount = torch.floor(actions[:, 0].view(-1, 1) * self.shares)
            # Number of shares to buy
            buy_amount = torch.floor(actions[:, 1].view(-1, 1) * self.balance / current_prices)
            
            # Apply transaction cost
            sell_cost = sell_amount * self.transaction_cost
            buy_cost = buy_amount * self.transaction_cost
            
            self.shares -= sell_amount
            self.balance += sell_amount*current_prices - sell_cost
            
            self.shares += buy_amount 
            self.balance -= buy_amount*current_prices + buy_cost
            
            self.balance_scaled = self.balance/self.balance_scale
            self.shares_scaled = self.shares/self.shares_scale
            
            # Reward: Portfolio value change
            portfolio_value = self.balance + (self.shares * current_prices)
            # reward = portfolio_value - (self.balance + sell_amount - sell_cost + buy_amount + buy_cost)
            self.reward = portfolio_value / self.reward_scale
            self.reward = self.reward.squeeze()
            
            self.elapsed_days += 1
            next_day = self.starting_day + self.elapsed_days
                
            # Check for out-of-bounds access            
            out_of_bounds = torch.where( (next_day) >= self.num_trading_days, 1, 0).squeeze()
            out_of_time = torch.where( self.elapsed_days >= self.max_trading_days, 1, 0).squeeze()
            out_of_money = torch.where( self.balance < self.min_balance, 1, 0).squeeze() 
            self.done = (out_of_bounds | out_of_time | out_of_money)

            for i in range(self.window_size):
                self.ticker_data[:, i*self.data_tensor_scaled.shape[1]:(i+1)*self.data_tensor_scaled.shape[1]] = self.data_tensor_scaled[ next_day.squeeze()+i-self.window_size, :]

            
            
            buf_states2 = self.state * torch.ones_like(self.state) # This should be some scaling values 

            [vals, probs] = ValueNet(buf_states2)
            
            self.buffer.update2(self.reward, buf_states2, self.done, vals)
    
            
            # env_ids = self.done.view(-1).nonzero(as_tuple=False).squeeze(-1)
    
            # if len(env_ids) > 0:
            #     self.reset_idx(env_ids)
        
                

    
    def render(self):
        """
        Render the environment state.
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares: {self.num_shares}")
        
    def unscale_data(self, scaled_values):
        """
        Unscale data using the fitted scaler.
        """
        return self.scaler.inverse_transform(scaled_values)

    def close(self):
        """
        Clean up resources.
        """
        pass

    def _get_price(self):
        current_day = self.starting_day + self.elapsed_days
        return self.data_tensor[current_day, self.column_indices['AAPL_Close']]
    
 

# #%% Initialize Environment
# from experience_buffer import Buffer

# data_path = "preprocessed_stock_data.csv"
# preprocessed_stock_data_df = pd.read_csv(data_path)

# # Set parameters
# initial_balance_range = [0, 10000]  # Initial cash balance
# transaction_cost = 0.001  # 0.1% transaction fee
# window_size = 4  # Number of past days to include in state

# # Initialize the trading environment
# env = TradingEnv(data=preprocessed_stock_data_df, 
#                  num_envs = 2, 
#                  initial_balance_range=initial_balance_range, 
#                  transaction_cost=transaction_cost, 
#                  window_size=window_size)

# buffer_horizon = 10
# replay_buffer = Buffer(buffer_horizon, env.num_envs, env.num_actions, env.num_states, 0.99)
# env.register_buffer(replay_buffer)
# print("Initial state:", env.state.shape)




# #%%
# import torch.nn as nn
# class Policy(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # define actor and critic networks
        
#         n_features = env.num_states
#         n_actions = env.num_actions
#         layer1_count = 256
#         layer2_count = 128
#         layer3_count = 64

        
#         self.shared1 = nn.Sequential(
#                                     nn.Linear(n_features, layer1_count),
#                                     nn.ELU()
#                                     )
        
#         self.shared2 = nn.Sequential(
#                                     nn.Linear(layer1_count+n_features, layer2_count),
#                                     nn.ELU()
#                                     )
        
        
#         self.policy1 = nn.Sequential(
#                                     nn.Linear(layer2_count+n_features, layer3_count),
#                                     nn.ELU()
#                                     )
#         self.policy2 = nn.Sequential(
#                                     nn.Linear(layer3_count+n_features, n_actions),
#                                     nn.Tanh(),
#                                     )
        
#         self.value1 = nn.Sequential(
#                                     nn.Linear(layer2_count+n_features, layer3_count),
#                                     nn.ELU()
#                                     )
#         self.value2 = nn.Sequential(
#                                     nn.Linear(layer3_count+n_features, 1),
#                                     )

#     def forward(self, x):
#         s1 = torch.cat((self.shared1(x), x), dim=-1)
#         s2 = torch.cat((self.shared2(s1), x), dim=-1)
#         v1 = torch.cat((self.value1(s2) , x), dim=-1)
#         v2 = self.value2(v1) 
#         p1 = torch.cat((self.policy1(s2), x), dim=-1)
#         p2 = self.policy2(p1)        
#         return v2, p2

# policy = Policy()

# #%% Test the Environment
# # Run a simple simulation loop


# reward_log = []
# env_2_watch = 0;
# for _ in range(10):
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     start_time = time.time()
    
    
#     actions = RandTensorRange( (env.num_envs, env.num_actions), 0.0, 1.0)    
#     log_probs = torch.ones((env.num_envs, env.num_actions))

#     print("balance pre step : ", env.balance[0])
#     print("shares pre step: ", env.shares[0])

#     env.step(actions, log_probs, policy)
    
#     print("balance post step : ", env.balance[0])
#     print("shares post step: ", env.shares[0])


#     # print(env.state[:, [0, env.data_tensor.shape[1], env.data_tensor.shape[1]*2, env.data_tensor.shape[1]*3 ]])
#     reward_log.append(env.reward[env_2_watch].detach().cpu().numpy())

#     elapsed_time = time.time() - start_time

    
#     [s1,a1,r1,s2,d, log_probs_old, returns] = env.buffer.get_SARS()
#     print("Env : ", env_2_watch, " Reward : ", env.reward[env_2_watch,...])
#     print('Elapsed Time : {}'.format(elapsed_time))
#     print('FPS : {}'.format(env.num_envs/elapsed_time))

#     # print('s1')
#     # print(s1[env_2_watch, ...])
#     # print('s2')
#     # print(s2[env_2_watch, ...])
#     print('a1')
#     print(a1[env_2_watch, ...])
#     print('r')
#     print(r1[env_2_watch, ...])
#     print('returns')
#     print(returns[env_2_watch, ...])
#     print('d')
#     print(d[env_2_watch, ...])
#     print('advantage')
#     print(env.buffer.rewards_to_go)
    
    

# #%% Clean Up
# env.close()

#%%
#     cp.render(0)
    

# # plt.figure(figsize=(9, 3))
# # plt.subplot(321)
# # plt.grid(True)
# # plt.plot(pos_log)
# # plt.subplot(322)
# # plt.plot(vel_log)
# # plt.grid(True)
# # plt.subplot(323)
# # plt.plot(theta_log)
# # plt.grid(True)
# # plt.subplot(324)
# # plt.plot(omega_log)
# # plt.grid(True)
# # plt.subplot(325)
# # plt.grid(True)
# # plt.plot(reward_log)


# #%%

# [s1,a1,r1,s2,d, lp_old, returns] = cp.buffer.get_SARS()
# print('s1')
# print(s1)
# print('s2')
# print(s2)
# print('a1')
# print(a1)
# print('r')
# print(r1)
# print('d')
# print(d)
# print('rewards_to_go')
# print(cp.buffer.rewards_to_go)

# #%%

# # [s1,a1,r1,s2,d] = cp.buffer.get_SARS_minibatch(6)
# # print('s1')
# # print(s1)
# # print('s2')
# # print(s2)
# # print('a1')
# # print(a1)
# # print('r')
# # print(r1)
# # print('d')
# # print(d)

# #%%

# dones_tmp = cp.buffer.d.clone()
# dones_tmp[:,0] = False
# dones_mask = torch.where(dones_tmp, 0, 1)
# dones_mask = torch.cumprod(dones_mask, dim=1)
# print(dones_mask)
