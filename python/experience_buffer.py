# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:11:15 2023

@author: Plutonium
"""

import torch

def RandTensorRange(size, min_val, max_val):
    tensor_range = max_val-min_val
    rt = torch.rand(size)*tensor_range + min_val
    return rt

class Buffer():
    def __init__(self, buf_horizon, num_envs, num_actions, num_states, gamma):
        self.buf_hor = buf_horizon
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.gamma_mask = torch.ones((self.buf_hor+1)) * self.gamma
        self.gamma_mask = torch.cumprod(self.gamma_mask, dim=0)/self.gamma
        
        # self.s1= torch.zeros([self.buf_len, self.num_states])
        # self.a = RandTensorRange([self.buf_len,], -1.0, 1.0)
        # self.r= torch.zeros([self.buf_len,])
        # self.s2 = torch.zeros([self.buf_len,self.num_states])
        # self.d = torch.zeros([self.buf_len,])==1
        
        self.s1 = torch.zeros([self.num_envs, self.buf_hor, self.num_states])
        self.a = torch.zeros([self.num_envs, self.buf_hor, self.num_actions])
        self.r = torch.zeros([self.num_envs, self.buf_hor,])
        self.s2 = torch.zeros([self.num_envs, self.buf_hor, self.num_states])
        self.d = torch.zeros([self.num_envs, self.buf_hor,]) == 1
        
        self.rewards_to_go = torch.zeros([self.num_envs, self.buf_hor,])
        self.value_gamma_scaler = torch.ones([self.num_envs, self.buf_hor,]) * self.gamma_mask[1:]
        
        self.log_probs = torch.zeros([self.num_envs, self.buf_hor, self.num_actions])
        
        self.value = torch.zeros([self.num_envs, self.buf_hor,])
        self.advantage = torch.zeros([self.num_envs, self.buf_hor,])
        self.returns = torch.zeros([self.num_envs, self.buf_hor,])
        
        self.gamma_vec = self.gamma**torch.linspace(0,self.buf_hor-1,self.buf_hor).reshape([1,self.buf_hor])
        
        
    def fill(self):
        pass
        # for i in range(self.buf_hor):
        #     actions = RandTensorRange([self.num_envs,], -1.0, 1.0)
        #     self.update(actions)
            
    def update1(self, s1, a1, lp):
        with torch.no_grad():
            self.s1 = self.s1.roll(1, 1)
            self.s1[:, 0, :] = s1
            
            self.a = self.a.roll(1, 1)
            self.a[:, 0, :] = a1
            
            self.log_probs = self.log_probs.roll(1, 1)
            self.log_probs[:, 0, :] = lp

            
    def update2(self, r2, s2, d2, val):
        with torch.no_grad():
            self.r = self.r.roll(1, 1)
            self.r[:, 0] = r2
            self.s2 = self.s2.roll(1, 1)
            self.s2[:, 0, :] = s2
            self.d = self.d.roll(1, 1)
            self.d[:, 0] = d2.view(-1)
            self.value = self.value.roll(1, 1)
            self.value[:, 0] = val.view(-1)
            
            
            self.active_mask = torch.cumsum(self.d, dim=1)==0
            self.active_mask1 = self.active_mask.detach().clone()
            self.active_mask1[:,0] = 1
            
            self.returns = self.returns.roll(dims=1, shifts=1)
            self.returns[:,0] = self.value[:,1]
            
            self.returns += (self.active_mask1*(self.r[:,0].view(-1,1) - 
                             self.value[:,1].view(-1,1)) + 
                             self.gamma*self.active_mask*self.value[:,0].view(-1,1))*self.gamma_vec
            
            
            # dones_tmp = self.d.clone()
            # dones_tmp[:,0] = False
            
            # dones_mask = torch.where(dones_tmp, 0, 1)
            # dones_mask = torch.cumprod(dones_mask, dim=1)
            
            
            # self.rewards_to_go = self.rewards_to_go.roll(1, 1)
            # self.rewards_to_go[:, 0] = 0
            # self.rewards_to_go += dones_mask*self.gamma_mask[0:self.buf_hor]*r2
            
            # dones_mask2 = torch.where(self.d, 0, 1)
            # dones_mask2 = torch.cumprod(dones_mask2, dim=1)
            # self.value_gamma_scaler = dones_mask2*self.gamma_mask[1:]
            
            # # returns = env.buffer.value_gamma_scaler*vals_s2[:,0] + r1

            
        
    def get_SARS(self):
        # return self.s1, self.a, self.r, self.s2, self.d 
        # return self.s1, self.a, self.rewards_to_go, self.s2, self.d, self.log_probs, self.returns
        return self.s1, self.a, self.r, self.s2, self.d, self.log_probs, self.returns
   
    def get_SARS_minibatch(self, num_samples):
        env_ids = torch.randint(low=0, high=self.num_envs, size=(num_samples,))
        # return self.s1[env_ids, :, :], self.a[env_ids, :], self.r[env_ids, :], self.s2[env_ids, :, :], self.d[env_ids, :] 
        return self.s1[env_ids, :, :], self.a[env_ids, :], self.rewards_to_go[env_ids, :], self.s2[env_ids, :, :], self.d[env_ids, :] 