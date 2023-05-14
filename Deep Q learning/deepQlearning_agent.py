import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random 
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque



# define the Q network
class Qnet(nn.Module):
    """
        Input of the Q-net should be either observation(new state) or the action or both. 
    """
    def __init__(self, input_dim, output_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, output_dim)  

    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x


# define the replay buffer
class ReplayBuffer():
    """
        Replay buffer is a list of transition. Each transition is a tuple of (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, transition):
        """ Save a transition"""
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]
        
    def sample(self, batch_size):
        """ Sample a batch of transition"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



# define the agent
class Agent():
    def __init__(self, env, device='cuda', learning_rate=0.001, discount_rate=0.8, epsilon=1, min_epsilon=0.01, epsilon_decay_rate=0.01):
        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.Q = Qnet(self.num_states, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(1000)
        self.score_values = []
    
    def explore(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state).float().to(self.device)
            action = torch.argmax(self.Q(state)).item()
            # action = action.cpu().detach().numpy() # convert tensor to numpy array
        else:
            action = self.env.action_space.sample()
        return action

    def remember(self, transition):
        self.replay_buffer.push(transition)

    def decay_epsilon(self):
        self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon)* np.exp(-self.epsilon_decay_rate) if self.epsilon > self.min_epsilon else self.min_epsilon
    

    def learn(self, batch_size):
        """
            Weigts of the network are updated in each epoch.
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        batch   = self.replay_buffer.sample(batch_size)
        states  = torch.tensor([transition[0] for transition in batch]).float().to(self.device)
        actions = torch.tensor([transition[1] for transition in batch]).to(self.device)
        rewards = torch.tensor([transition[2] for transition in batch]).float().to(self.device)
        next_states = torch.tensor([transition[3] for transition in batch]).float().to(self.device)
        dones = torch.tensor([transition[4] for transition in batch]).to(self.device)

        # calculate the loss
        current_q_values = self.Q(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.Q(next_states).max(1)[0].detach()
        target_q_values = rewards + self.discount_rate * next_q_values * (1 - dones.float())
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # update the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, number_of_epochs, batch_size):
        for epoch in range(number_of_epochs):
            state = self.env.reset()[0]
            done = False
            score = 0
            start_time = time.time()
            while not done:
                action = self.explore(state)
                next_state, reward, done, _, _ = self.env.step(action)
                score += reward  
                self.remember((state, action, reward, next_state, done))
                self.learn(batch_size)
                state = next_state
                self.env.render()
                current_time = time.time()
                if current_time - start_time > 60:
                    break
                
            self.decay_epsilon()
            print('Epoch: ', epoch+1, ' Epsilon: ', self.epsilon, ' Score: ', score)
            self.score_values.append(score)

    def plot_score(self):
        plt.plot(np.arange(len(self.score_values)), self.score_values)
        plt.ylabel('Score')
        plt.xlabel('Epochs')
        plt.show()
        

