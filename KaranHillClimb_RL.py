#!/usr/bin/env python
# coding: utf-8

# In[16]:


import gym
import random
import numpy as np


# In[17]:


class HCAgent(): 
    def __init__(self, environment):
        self.state_dim = environment.observation_space.shape
        self.action_size = environment.action_space.n
        self.build()
        
    def build(self):
        # Initialize weights matrix(4x2), reward and noise
        self.weights = 1e-4*np.random.rand(*self.state_dim, self.action_size)
        self.best_reward = -np.Inf
        self.best_weights = np.copy(self.weights)
        self.noise_scale = 1e-2
        
    def getActions(self, state):
        # dot product of state matrix(1x4) and weights
        p = np.dot(state, self.weights)
        #returns which action to perform
        action = np.argmax(p)
        return action
    
    def updateModel(self, reward):
        # update best_reward with reward if value is higher
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            #reduce noise 
            self.noise_scale = max(self.noise_scale/2, 1e-3)
        else:
            #increase noise to explore other regions
            self.noise_scale = min(self.noise_scale*2, 2)
        #Update weights    
        self.weights = self.best_weights + self.noise_scale * np.random.rand(*self.state_dim, self.action_size)


# In[18]:


# Initialize the "Cart-Pole" environment
environment = gym.make('CartPole-v0')
print("Observation space:", environment.observation_space)
print("Action space:", environment.action_space)

agent = HCAgent(environment)
# setting total episodes to iterate to 50 episodes
TOTAL_EPISODES = 50

# To calculate episodes crossing target value
episodeWithExcessiveReward = []
rewardThresholdValue = 200

# Calculating total reward for each episodes
for episode in range(TOTAL_EPISODES):
    state = environment.reset()
    total_reward = 0
    status = False
    while not status:
        act = agent.getActions(state)
        state, reward, status, info = environment.step(act)
        environment.render()
        total_reward += reward
    
    # printing episodes with rewards earned respectively
    agent.updateModel(total_reward)
    print("Episode Number : {}, Rewards(Total): {:.2f}".format(episode+1, total_reward))
    if int(total_reward) >= rewardThresholdValue:
        episodeWithExcessiveReward.append(episode+1)
        
# printing episode numbers with reward greater than target value
if len(episodeWithExcessiveReward) > 0:
    episodesStr = ', '.join(str(x) for x in episodeWithExcessiveReward)
    print("Episode(s) exceeding target reward : "+episodesStr)
else:
    # if none of the episodes reach target value
    rewardThresholdValueStr = str(rewardThresholdValue)
    print("None of any episode's rewards exceeds "+rewardThresholdValueStr)


# In[ ]:




