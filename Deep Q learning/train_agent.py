import gym
import torch
from deepQlearning_agent import Agent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='LunarLander-v2', help='the name of the game')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')

game = parser.parse_args().game

env = gym.make(game, render_mode='human')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
learning_rate       = 0.001
discount_rate       = 0.8
epsilon             = 1
min_epsilon         = 0.01
epsilon_decay_rate  = 0.01
number_of_epochs    = parser.parse_args().epochs
batch_size          = 128


agent = Agent(env, device, learning_rate, discount_rate, epsilon, min_epsilon, epsilon_decay_rate)
agent.train(number_of_epochs, batch_size)
env.close()

agent.plot_score() 

# save the model
# agent.save_model('Deep Q learning\pretrained-models\DQL-{}.pth'.format(game))