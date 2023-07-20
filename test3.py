import tensorflow as tf
import numpy as np
import gym
import math
from PIL import Image
import pygame, sys
from pygame.locals import *
from tensorflow import keras
from FRCGameEnv2 import env

environment = env()

input_shape = 15
num_actions = 3


policy_network_red_vel_x = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])
policy_network_red_vel_y = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])
policy_network_red_vel_a= tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])

policy_network_blue_vel_x = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])
policy_network_blue_vel_y = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])
policy_network_blue_vel_a= tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(201, activation='softmax')
])

# Set up the optimizer and loss function
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Set up lists to store episode rewards and lengths
episode_rewards = []
episode_lengths = []

num_episodes = 50
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
            action_probs_vel_x, action_probs_vel_y, action_probs_vel_a = 0, 0, 0
            if agent.startswith('red'):
                with tf.device('/GPU:0'):
                    action_probs_vel_x = policy_network_red_vel_x.predict(np.array([state[agent]]))[0]
                    action_probs_vel_y = policy_network_red_vel_y.predict(np.array([state[agent]]))[0]
                    action_probs_vel_a = policy_network_red_vel_a.predict(np.array([state[agent]]))[0]
            else:
                with tf.device('/GPU:0'):
                    action_probs_vel_x = policy_network_blue_vel_x.predict(np.array([state[agent]]))[0]
                    action_probs_vel_y = policy_network_blue_vel_y.predict(np.array([state[agent]]))[0]
                    action_probs_vel_a = policy_network_blue_vel_a.predict(np.array([state[agent]]))[0]

            # Choose an action based on the action probabilities
            action_vel_x = np.random.choice(201, p=action_probs_vel_x)
            action_vel_y = np.random.choice(201, p=action_probs_vel_y)
            action_vel_a = np.random.choice(201, p=action_probs_vel_a)

            holder = []

            for action in [action_vel_x, action_vel_y, action_vel_a]:
                if action > 101:
                    holder.append((action - 101) / 100)
                elif action < 101:
                    holder.append(-(action / 100))
                else:
                    holder.append(0)

            action_holder[agent] = [action_vel_x, action_vel_y, action_vel_a]
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
    red_discounted_rewards = list(map(int, red_discounted_rewards))
    red_discounted_rewards -= np.mean(red_discounted_rewards)
    if all(red_discounted_rewards) != 0:
        red_discounted_rewards /= np.std(red_discounted_rewards)


    blue_discounted_rewards = list(map(int, blue_discounted_rewards))
    blue_discounted_rewards -= np.mean(blue_discounted_rewards)
    if all(blue_discounted_rewards) != 0:
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
        red_discounted_rewards = [tf.convert_to_tensor([red_discounted_rewards, red_discounted_rewards, red_discounted_rewards])]
        blue_discounted_rewards = [tf.convert_to_tensor([blue_discounted_rewards, blue_discounted_rewards, blue_discounted_rewards])]


    # Train the policy network using the REINFORCE algorithm
    for policy_network in [policy_network_red_vel_x, policy_network_red_vel_y, policy_network_red_vel_a]:
        with tf.GradientTape() as tape:
            # Get the action probabilities from the policy network
            action_probs = policy_network(states['red_1'])
            # Calculate the loss
            loss = tf.cast(tf.math.log(tf.gather(action_probs, np.asarray(actions['red_1'], dtype=np.int32), axis=1, batch_dims=1)), tf.float64)

            loss = loss @ red_discounted_rewards[0]
            loss = -tf.reduce_sum(loss)

        # Calculate the gradients and update the policy network
        grads = tape.gradient(loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        # Store the episode reward and length
        episode_rewards.append(episode_rewards)
        episode_lengths.append(episode_length)

        policy_network.save(
            f'keras/{str(["policy_network_red_vel_x", "policy_network_red_vel_y", "policy_network_red_vel_a"][[policy_network_red_vel_x, policy_network_red_vel_y, policy_network_red_vel_a].index(policy_network)])}/')

    for policy_network in [policy_network_blue_vel_x, policy_network_blue_vel_y, policy_network_blue_vel_a]:
        with tf.GradientTape() as tape:
            # Get the action probabilities from the policy network
            action_probs = policy_network(states['blue_1'])
            # Calculate the loss
            loss = tf.cast(tf.math.log(tf.gather(action_probs, np.asarray(actions['blue_1'], dtype=np.int32), axis=1, batch_dims=1)), tf.float64)

            loss = loss @ blue_discounted_rewards[0]
            loss = -tf.reduce_sum(loss)

        # Calculate the gradients and update the policy network
        grads = tape.gradient(loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        # Store the episode reward and length
        episode_rewards.append(episode_rewards)
        episode_lengths.append(episode_length)

        policy_network.save(f'keras/{str(["policy_network_blue_vel_x", "policy_network_blue_vel_y", "policy_network_blue_vel_a"][[policy_network_blue_vel_x, policy_network_blue_vel_y, policy_network_blue_vel_a].index(policy_network)])}/')