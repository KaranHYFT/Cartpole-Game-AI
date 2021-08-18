#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import gym
import random
import math


# In[5]:


# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')

# No. of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x_dot, theta, theta_dot')
# No. of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# set position between -2.4 to +2.4
STATE_BOUNDS[1] = [-2.4, 2.4]
# set angle between -15 to +15
STATE_BOUNDS[3] = [-math.radians(15), math.radians(15)]
# Index of the action

# Q table creation
Qtable = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# constant for learning and exploration
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

# constants for simulation
TOTAL_EPISODES = 200
# Iterate 1 episode for MAX_T Time
MAX_T = 250
# No. of times for which pole will not fall
SOLVED_T = 200


def get_exploreRate(t):
    #Logrithmic decaying explore rate
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25.0)))    

def get_learningRate(t):
    #Logrithmic decaying learning rate
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25.0))) 


def choose_action(state, exploreRate):
    
    if random.random() < exploreRate:
        # Select a random action
        action = env.action_space.sample()
    else:
        # Select action having highest Q
        action = np.argmax(Qtable[state])
    return action

#convert state values into descrete value
def state_to_bucket(state):
    bucketList = []
    for i in range(len(state)):
        # if state value is less than lower range 
        if state[i] <= STATE_BOUNDS[i][0]:
            val = 0
        # if state value is more than high range     
        elif state[i] >= STATE_BOUNDS[i][1]:
            val = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            boundRange = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/boundRange
            scaling = (NUM_BUCKETS[i]-1)/boundRange
            val = int(round(scaling*state[i] - offset))
        bucketList.append(val)
    return tuple(bucketList)


# In[8]:


def QAgent():

    # call to get leraning and explore rate
    learningRate = get_learningRate(0)
    exploreRate = get_exploreRate(0)
    discountfactor = 0.99
    #string to print successfull Episodes
    successfulEp =""
    # to check wheter it meet our requirement (SOLVED = True ,when episode completed after 200 successful time step)
    SOLVED = False 
    
    for episode in range(TOTAL_EPISODES):
        SOLVED = False
        INSIDE_DONE = False
        # Reset the environment
        observation = env.reset()

        # the initial state
        state0 = state_to_bucket(observation)

        for t in range(MAX_T):
            # simulate
            env.render()
            # Select an action
            action = choose_action(state0, exploreRate)
            # Execute the action
            observation, Reward, done, _ = env.step(action)
            # Observe the result
            state = state_to_bucket(observation)
            # update the Q Table
            bestQ = np.amax(Qtable[state])
            Qtable[state0 + (action,)] += learningRate*(Reward + discountfactor*(bestQ) - Qtable[state0 + (action,)])
            # Assigning current state for next state
            state0 = state
            
            if done:
               print("Episode %d finished after %f time steps" % (episode, t))
               INSIDE_DONE = True
               # when pole does not fall for more than 200 time step
               if (t >= SOLVED_T):
                   SOLVED = True
               break
               
        if SOLVED:
            successfulEp += "\nEpisode %d finished after %f time steps" % (episode, t)
        elif INSIDE_DONE == False:
            print("Episode %d finished after %f time steps" % (episode, t))
            successfulEp += "\nEpisode %d finished after %f time steps" % (episode, t)
            
        # Update learning and exploration rate
        exploreRate = get_exploreRate(episode)
        learningRate = get_learningRate(episode)

    
    print("\n------successful Episodes--------")            
    print(successfulEp)        


if __name__ == "__main__":
    QAgent()


# In[ ]:




