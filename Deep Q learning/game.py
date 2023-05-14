import gym 
import random
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')

# Hyper parameters
episodes = 100



# the agent acts randomly
for episode in range(1, episodes+1):
   state = env.reset()[0]
   print('state ', state)
   done = False
   score = 0
   while not done:
      action = random.choice([0,1])
      observation, reward, done, _, _ = env.step(action)
      score += reward   
      env.render()

      # print('Episode:{} Score:{} State: {}'.format(episode, score, observation))

env.close()