import numpy as np
import pandas as pd
from scipy.integrate import odeint
import tensorflow.compat.v2 as tf
from tensorflow import keras
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

epsilon = 10e-2
x0 = np.array([3.14, 4.58, 1])
xd = np.array([5.137, 4.033, 0])

class microbial_3_species():
    def __init__(self):
        self.initial_state = np.array([3.14, 4.58, 1])
        self.desired_state = np.array([5.137, 4.033, 0])
        self.epsilon = epsilon
        self.df_action = pd.DataFrame(['increase', 'no_change', 'decrease'],
                                      columns=['3'])
        pass

    def System_Dynamics(self, x: list, t):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        dx1dt = 0.1 + x1 * (1 - x1/5) * (x1/3 - 1) - 0.1 * x1 * x3 / (1 + x3)
        dx2dt = 0.1 + x2 * (1 - x2/4) * (x2 - 1) + x2 * x3 / (1 + x3)
        dx3dt = x3 * (1 - x3/2) * (x3 - 1)
        return [dx1dt, dx2dt, dx3dt]

    def find(self, state: list):
        reach_desired_state = False
        state_ = np.array(state)
        if np.linalg.norm(state_ - xd, np.inf) < self.epsilon:
            reach_desired_state = True
        return reach_desired_state

    def check_range(self, state):
        in_range = True
        for i in range(3):
            if state[i] < 0 or state[i] > 6:
                in_range = False
        return in_range

    def action_to_num(self, action):
        """
        Convert action from string to number.
        :param action: takes form of 'increase', 'no_change', 'decrease'
        :return: action number
        """
        for i in range(len(self.df_action.index)):
            action_i = self.df_action.loc[i]['3']
            if action == action_i:
                return i

    def num_to_action(self, num):
        action = self.df_action['3'][num]
        return action

    def reward_func(self, state: list):
        state_ = np.array(state)
        xd = np.array(self.desired_state)
        reward = - (np.linalg.norm(state_ - xd)) / 10
        return reward

    def step(self, state: list, action_num):
        action = self.num_to_action(action_num)
        current_state = state

        t = np.linspace(0, 1, 20)
        state_after_action = current_state
        if action == 'increase':
            state_after_action[2] += 0.2
        if action == 'decrease':
            state_after_action[2] -= 0.2
        if action == 'no_change':
            state_after_action = current_state
        feedback = odeint(self.System_Dynamics, state_after_action, t)
        next_state = feedback[20 - 1]
        if abs(self.reward_func(next_state) - self.reward_func(current_state)) < 0.001:
            return next_state, -1000, True

        if not self.check_range(next_state):
            return next_state, -1000, True

        if self.find(next_state):
            return next_state, 1000, True
        else:
            return next_state, self.reward_func(next_state), False

    def reset(self):
        state = np.array([3.14, 4.58, 1])
        return state

env = microbial_3_species()

class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:  # Handle collisions
            return hash(codeword) % self.features
        self.codebook[codeword] = count
        return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) /
                    self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

class SARSAAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                 learning_rate=0.03, epsilon=0.01):
        self.action_n = 3  # Number of actions
        self.obs_low = np.array([0, 0, 0])
        self.obs_scale = np.array([6, 6, 6])  # Range of observation space
        self.encoder = TileCoder(layers, features)  # Tile coder
        self.w = np.zeros(features)  # Weights
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate  # Learning rate
        self.epsilon = epsilon  # Exploration rate

    def encode(self, observation, action):  # Encoding function
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):  # Q-value calculation
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation):  # Decide action
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward,
              next_observation, done, next_action):  # Learning
        u = reward + (1. - done) * self.gamma * self.get_q(next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)

agent = SARSAAgent(env)

def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = np.array([3.14, 4.58, 1])
    action = agent.decide(observation)
    num_iter = 0
    action_path = []
    reward_path = []
    while True:
        num_iter += 1
        if render:
            env.render()
        next_observation, reward, done = env.step(observation, action)
        episode_reward += reward
        reward_path.append(reward)
        if len(reward_path) > 2:
            if abs(reward_path[-1] - reward_path[-2]) < 0.001:
                reward = -1000
                done = True
        next_action = agent.decide(next_observation)  # Not relevant if terminal state
        action_path.append(action)
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            if episode_reward > 800:
                print(action_path)
            break
        observation, action = next_observation, next_action
        if num_iter > 1000:
            break
    return episode_reward

episodes = 1000
episode_rewards = []
iter_num = 0
for episode in range(episodes):
    iter_num += 1
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)

def test_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    num_iter = 0
    action_path = []
    df_test = pd.DataFrame(columns=['state', 'action', 'reward', 'done'])
    for i in range(1000):
        next_observation, reward, done = env.step(observation, action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # Not relevant if terminal state
        action_path.append(action)
        df_test.loc[i] = [observation, action, reward, done]
        if done:
            break
        observation, action = next_observation, next_action
    return df_test

action_path_1 = [2, 1, 1, 1, 1]

# Simulate the system based on the action path of action_path_1
state = env.reset()
state_path_1 = []
for i in range(len(action_path_1)):
    state_path_1.append(state)
    state, _, _ = env.step(state, action_path_1[i])

# Simulate more steps with no actions
for i in range(30):
    state_path_1.append(state)
    state, _, _ = env.step(state, 1)

state_path_1 = np.round(state_path_1, 2)

# Extract x1, x2, and x3 values from state_path_1
x1, x2, x3 = [], [], []
for i in range(len(state_path_1)):
    x1.append(state_path_1[i][0])
    x2.append(state_path_1[i][1])
    x3.append(state_path_1[i][2])

# Plot x1, x2, and x3 in 2D
fig, ax = plt.subplots()
ax.plot(x1, label='x1')
ax.plot(x2, label='x2')
ax.plot(x3, label='x3')
ax.legend()

# Set axis limits
ax.set_xlim([0, 30])
ax.set_ylim([-1, 6])
plt.show()