import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
#%%
class TradingEnv(gym.Env):
    def __init__(self, data, num_envs=1, initial_balance_range=[0, 10000], transaction_cost=0.001, window_size=10):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.column_indices = {name: idx for idx, name in enumerate(self.data.columns)}
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        with open("scaler_sb.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        self.data_tensor = torch.tensor(self.data.values, dtype=torch.float32)
        self.data_tensor_scaled = torch.tensor(self.scaled_data, dtype=torch.float32)
        

        self.initial_balance_range = initial_balance_range
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        self.num_actions = 2  # Sell %, Buy %
        self.num_states = self.data_tensor_scaled.shape[1] * self.window_size + 2  # Features + balance + shares

        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32
        )
        
        self.balance_scale = 1e6
        self.shares_scale = 1e6
        self.reward_scale = 1e6
        self.max_trading_days = 365 # Lets say 1 year of trading
        self.num_trading_days = self.data_tensor.shape[0]
        
        self.reset()

    def reset(self, seed = None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        
        self.balance = np.random.uniform(*self.initial_balance_range)
        self.balance_scaled = self.balance/self.balance_scale
        
        self.shares = 0
        self.shares_scaled = self.shares/self.shares_scale
        
        self.current_step = np.random.randint(self.window_size, len(self.data) - 365)
        self.done = False

        self.state = self._get_state()
        
        self.elapsed_days = 0
        
        return self.state, {}

    def step(self, action):
        sell_amount = action[0] * self.shares
        buy_amount = action[1] * self.balance / self._get_price()
    
        # Transaction costs
        sell_cost = sell_amount * self.transaction_cost
        buy_cost = buy_amount * self.transaction_cost
    
        # Update portfolio
        self.shares -= sell_amount
        self.balance += sell_amount * self._get_price() - sell_cost
        self.shares += buy_amount
        self.balance -= buy_amount * self._get_price() + buy_cost
        

    
        # Reward: Portfolio value change
        portfolio_value = self.balance + self.shares * self._get_price()
        reward = float(portfolio_value)/self.reward_scale  # Explicitly cast to float
    
        # Advance step
        self.current_step += 1
        self.elapsed_days += 1
        
        out_of_bounds = self.current_step >= self.num_trading_days
        out_of_time = self.elapsed_days >= self.max_trading_days
        out_of_money = self.balance < 0 
        
        self.done = out_of_bounds or out_of_time or out_of_money
        terminated = bool(self.done)
        truncated = False
    
        self.state = self._get_state()
        
        return self.state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        print(f"Balance: {self.balance}, Shares: {self.shares}")

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step
        ticker_data = self.data_tensor_scaled[start:end].flatten()
        
        self.balance_scaled = self.balance/self.balance_scale
        self.shares_scaled = self.shares/self.shares_scale
        
        # Explicitly cast the state to float32
        return np.concatenate([ticker_data, [self.balance_scaled, self.shares_scaled]]).astype(np.float32)

    def _get_price(self):
        return self.data_tensor[self.current_step, self.column_indices['AAPL_Close']]
    

    
#%% Validate and test the environment
data_path = "preprocessed_stock_data.csv"
preprocessed_stock_data_df = pd.read_csv(data_path)

env = TradingEnv(data=preprocessed_stock_data_df)
check_env(env)  # Validate Gymnasium API compliance

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

model.learn(total_timesteps=1000000)

#%%
# Save the trained model
model.save("ppo_trading_model")

#%%
plt.close('all')
# Load the trained model
trained_model_path = "ppo_trading_model"
model = PPO.load(trained_model_path)

# Reset the environment
obs, info = env.reset()
done = False
total_reward = 0
portfolio_values = []
actions_taken = []

# Run one episode
env.max_trading_days = 1000
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward
    portfolio_value = env.balance + env.shares * env._get_price()
    portfolio_values.append(portfolio_value)
    actions_taken.append(action)  # Record actions

    env.render()  # Optional: Display the current state

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
obs, info = env.reset()
print("Observation shape:", obs.shape)
print("Observation dtype:", obs.dtype)
print("Observation:", obs)

print("Initial balance:", env.balance)
print("Initial shares:", env.shares)

print("State from reset:", obs)
print("State length:", len(obs))

for action in [[0.0, 0.0], [0.5, 0.5], [0.5, 1.0]]:
    print("-"*20)
    print(f"Action: {action}")
    print("Sell Amount:", action[0] * env.shares)
    print("Buy Amount:", action[1] * env.balance / env._get_price())
    print("Current Price:", env._get_price())
    print("")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"New Balance: {env.balance}, New Shares: {env.shares}")
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    print("")    
    portfolio_value = env.balance + env.shares * env._get_price()
    print(f"Portfolio Value: {portfolio_value}, Reward: {reward}")
    
    
