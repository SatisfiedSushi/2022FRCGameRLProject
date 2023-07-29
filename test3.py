import tensorflow as tf
import numpy as np
import gym
import math
from PIL import Image
import pygame, sys
from pygame.locals import *
from tensorflow import keras
from FRCGameEnv2 import env
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

environment = env()

input_shape = 15
num_actions = 26

'''policy_network_red_vel_x = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
policy_network_red_vel_y = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
policy_network_red_vel_a= tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

policy_network_blue_vel_x = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
policy_network_blue_vel_y = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
policy_network_blue_vel_a = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])'''

'''
0 -> move forward
1 -> move backward
2 -> move left
3 -> move right
4 -> move up+right
5 -> move up+left
6 -> move back+right
7 -> move back+left
8 -> turn left
9 -> turn right
10 -> move forward+turn left
11 -> move backward+turn left
12 -> move left+turn left
13 -> move right+turn left
14 -> move up+right+turn left
15 -> move up+left+turn left
16 -> move back+right+turn left
17 -> move back+left+turn left
18 -> move forward+turn right
19 -> move backward+turn right
20 -> move left+turn right
21 -> move right+turn right
22 -> move up+right+turn right
23 -> move up+left+turn right
24 -> move back+right+turn right
25 -> move back+left+turn right
'''

policy_network_red = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

policy_network_blue = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

'''policy_network_red = tf.keras.models.load_model('keras/policy_network_red')
policy_network_blue = tf.keras.models.load_model('keras/policy_network_blue')'''

'''policy_network_blue_vel_x = tf.keras.models.load_model('keras/policy_network_blue_vel_x')
policy_network_blue_vel_y = tf.keras.models.load_model('keras/policy_network_blue_vel_y')
policy_network_blue_vel_a = tf.keras.models.load_model('keras/policy_network_blue_vel_a')
policy_network_red_vel_x = tf.keras.models.load_model('keras/policy_network_red_vel_x')
policy_network_red_vel_y = tf.keras.models.load_model('keras/policy_network_red_vel_y')
policy_network_red_vel_a = tf.keras.models.load_model('keras/policy_network_red_vel_a')'''

# Set up the optimizer and loss function
optimizer_red = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
loss_fn_red = tf.keras.losses.SparseCategoricalCrossentropy()

# Set up lists to store episode rewards and lengths
episode_rewards_red = []
episode_lengths_red = []

# Set up the optimizer and loss function
optimizer_blue = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
loss_fn_blue = tf.keras.losses.SparseCategoricalCrossentropy()

# Set up lists to store episode rewards and lengths
episode_rewards_blue = []
episode_lengths_blue = []

num_episodes = 5
discount_factor = 0.99

# Train the agent using the REINFORCE algorithm
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = environment.reset()[0]
    red_episode_reward = 0
    blue_episode_reward = 0
    episode_length = 0

    # Keep track of the states, actions, and rewards for each step in the episode
    states = []
    actions = []
    red_rewards = []
    blue_rewards = []

    # Run the episode
    while True:
        action_holder = {}
        action_holder_converted = {}
        for agent in environment.agents:
            action_probs = 0
            if agent.startswith('red'):
                with tf.device('/GPU:0'):
                    action_probs = policy_network_red.predict(np.array([state[agent]]))[0]
            else:
                with tf.device('/GPU:0'):
                    action_probs = policy_network_blue.predict(np.array([state[agent]]))[0]

            # Choose an action based on the action probabilities
            # action = np.random.choice(26, p=action_probs)
            action = max(action_probs)

            match action:
                case 0:
                    holder = [0, 1, 0] # move forward
                case 1:
                    holder = [0, -1, 0] # move backward
                case 2:
                    holder = [-1, 0, 0] # move left
                case 3:
                    holder = [1, 0, 0] # move right
                case 4:
                    holder = [1, 1, 0] # move up+right
                case 5:
                    holder = [-1, 1, 0] # move up+left
                case 6:
                    holder = [1, -1, 0] # move back+right
                case 7:
                    holder = [-1, -1, 0] # move back+left
                case 8:
                    holder = [0, 0, -1] # turn left
                case 9:
                    holder = [0, 0, 1] # turn right
                case 10:
                    holder = [0, 1, -1] # move forward+turn left
                case 11:
                    holder = [0, -1, -1] # move backward+turn left
                case 12:
                    holder = [-1, 0, -1] # move left+turn left
                case 13:
                    holder = [1, 0, -1] # move right+turn left
                case 14:
                    holder = [1, 1, -1] # move up+right+turn left
                case 15:
                    holder = [-1, 1, -1] # move up+left+turn left
                case 16:
                    holder = [1, -1, -1] # move back+right+turn left
                case 17:
                    holder = [-1, -1, -1] # move back+left+turn left
                case 18:
                    holder = [0, 1, 1] # move forward+turn right
                case 19:
                    holder = [0, -1, 1] # move backward+turn right
                case 20:
                    holder = [-1, 0, 1] # move left+turn right
                case 21:
                    holder = [1, 0, 1] # move right+turn right
                case 22:
                    holder = [1, 1, 1] # move up+right+turn right
                case 23:
                    holder = [-1, 1, 1] # move up+left+turn right
                case 24:
                    holder = [1, -1, 1] # move back+right+turn right
                case 25:
                    holder = [-1, -1, 1] # move back+left+turn right
                case _:
                    holder = [0, 0, 0] # do nothing


            action_holder[agent] = action
            action_holder_converted[agent] = holder


        # Take the chosen action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = environment.step(action_holder_converted)

        # Store the current state, action, and reward
        states.append(state)
        actions.append(action_holder)
        red_rewards.append(reward['red_1'])
        blue_rewards.append(reward['blue_1'])

        # Update the current state and episode reward
        state = next_state
        red_episode_reward += reward['red_1']
        blue_episode_reward += reward['blue_1']
        episode_length += 1
        # End the episode if the environment is done
        if truncated['__all__']:
            print(f'red episode reward: {red_episode_reward}')
            print(f'blue episode reward: {blue_episode_reward}')
            print('Episode {} done'.format(episode + 1))
            break
        # environment.render()
    # Calculate the discounted rewards for each step in the episode
    red_discounted_rewards = np.zeros_like(red_rewards)
    blue_discounted_rewards = np.zeros_like(blue_rewards)
    running_total = 0
    for i in reversed(range(len(red_rewards))):
        running_total = running_total * discount_factor + red_rewards[i]
        red_discounted_rewards[i] = running_total

    running_total = 0
    for i in reversed(range(len(blue_rewards))):
        running_total = running_total * discount_factor + blue_rewards[i]
        blue_discounted_rewards[i] = running_total

    # Normalize the discounted rewards
    # red_discounted_rewards = [list(map(int, red_discounted_rewards))]
    red_discounted_rewards -= np.mean(red_discounted_rewards)
    if np.asarray(red_discounted_rewards).all() != 0:
        red_discounted_rewards /= np.std(red_discounted_rewards)


    # blue_discounted_rewards = [list(map(int, blue_discounted_rewards))]
    blue_discounted_rewards -= np.mean(blue_discounted_rewards)
    if np.asarray(blue_discounted_rewards).all() != 0:
        blue_discounted_rewards /= np.std(blue_discounted_rewards)

    # Convert the lists of states, actions, and discounted rewards to tensors
    '''states = {agent: states[states.index(state)][agent] for state in states for agent in ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]}
    states = {agent: tf.convert_to_tensor(states[agent]) for agent in ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]}'''
    blue1 = []
    blue2 = []
    blue3 = []
    red1 = []
    red2 = []
    red3 = []

    for state in states:
        blue1.append(state['blue_1'])
        blue2.append(state['blue_2'])
        blue3.append(state['blue_3'])
        red1.append(state['red_1'])
        red2.append(state['red_2'])
        red3.append(state['red_3'])
    with tf.device('/GPU:0'):
        states = {'blue_1': tf.convert_to_tensor(blue1), 'blue_2': tf.convert_to_tensor(blue2), 'blue_3': tf.convert_to_tensor(blue3), 'red_1': tf.convert_to_tensor(red1), 'red_2': tf.convert_to_tensor(red2), 'red_3': tf.convert_to_tensor(red3)}

    blue1 = []
    blue2 = []
    blue3 = []
    red1 = []
    red2 = []
    red3 = []
    for action in actions:
        blue1.append(action['blue_1'])
        blue2.append(action['blue_2'])
        blue3.append(action['blue_3'])
        red1.append(action['red_1'])
        red2.append(action['red_2'])
        red3.append(action['red_3'])
    with tf.device('/GPU:0'):
        actions = {'blue_1': tf.convert_to_tensor(blue1), 'blue_2': tf.convert_to_tensor(blue2), 'blue_3': tf.convert_to_tensor(blue3), 'red_1': tf.convert_to_tensor(red1), 'red_2': tf.convert_to_tensor(red2), 'red_3': tf.convert_to_tensor(red3)}
        red_discounted_rewards = tf.convert_to_tensor(red_discounted_rewards)
        blue_discounted_rewards = tf.convert_to_tensor(blue_discounted_rewards)

    # Train the policy network using the REINFORCE algorithm
    for policy_network in [policy_network_red]:
        with tf.GradientTape() as tape:
            # Get the action probabilities from the policy network
            action_probs = policy_network(states['red_1'])
            # Calculate the loss
            loss = tf.cast(tf.math.log(tf.gather(action_probs, np.asarray(actions['red_1'], dtype=np.int32), axis=1, batch_dims=1)), tf.float64)

            loss = loss * red_discounted_rewards
            loss = -tf.reduce_sum(loss)

        # Calculate the gradients and update the policy network
        grads_red = tape.gradient(loss, policy_network.trainable_variables)
        optimizer_red.apply_gradients(zip(grads_red, policy_network.trainable_variables))

        # Store the episode reward and length
        episode_rewards_red.append(episode_rewards_red)
        episode_lengths_red.append(episode_length)

        policy_network.save(
            f'keras/policy_network_red/')

    for policy_network in [policy_network_blue]:
        with tf.GradientTape() as tape:
            # Get the action probabilities from the policy network
            action_probs = policy_network(states['blue_1'])
            # Calculate the loss
            loss = tf.cast(tf.math.log(tf.gather(action_probs, np.asarray(actions['blue_1'], dtype=np.int32), axis=1, batch_dims=1)), tf.float64)

            loss = loss * blue_discounted_rewards
            loss = -tf.reduce_sum(loss)

        # Calculate the gradients and update the policy network
        grads_blue = tape.gradient(loss, policy_network.trainable_variables)
        optimizer_blue.apply_gradients(zip(grads_blue, policy_network.trainable_variables))

        # Store the episode reward and length
        episode_rewards_blue.append(episode_rewards_blue)
        episode_lengths_blue.append(episode_length)

        policy_network.save(f'keras/policy_network_blue/')