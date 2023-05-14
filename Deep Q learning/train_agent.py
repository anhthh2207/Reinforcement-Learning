import gym
import torch
from deepQlearning_agent import Agent

env = gym.make('CartPole-v1', render_mode='human')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
learning_rate       = 0.001
discount_rate       = 0.8
epsilon             = 1
min_epsilon         = 0.01
epsilon_decay_rate  = 0.01
number_of_epochs    = 500
batch_size          = 128


agent = Agent(env, device, learning_rate, discount_rate, epsilon, min_epsilon, epsilon_decay_rate)
agent.train(number_of_epochs, batch_size)
env.close()

agent.plot_score()