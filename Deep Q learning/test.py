import gym
from deepQlearning_agent import Agent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='LunarLander-v2', help='the name of the game')

game = parser.parse_args().game
env = gym.make(game)
env.reset()
agent = Agent(env, device='cpu')
agent.load_model('Deep Q learning\pretrained-models\DQL-{}.pth'.format(game))

epochs = 10
for epoch in range(epochs):
    state = env.reset()[0]
    done = False
    score = 0
    while not done:
        env.render()
        action = agent.explore(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
        env.render()
    print('Epoch: {} Score: {}'.format(epoch, score))
env.close()