#pygame essentials
import time

import numpy as np
from FRCGameEnv2 import env
import tensorflow as tf

#openai gym env
environment = env()
input_shape = 15
num_actions = 3
state = environment.reset()[0]

done = False
count=0
steps = 0
#loading trained model
policy_network_blue_vel_x = tf.keras.models.load_model('keras/policy_network_blue_vel_x')
policy_network_blue_vel_y = tf.keras.models.load_model('keras/policy_network_blue_vel_y')
policy_network_blue_vel_a = tf.keras.models.load_model('keras/policy_network_blue_vel_a')
policy_network_red_vel_x = tf.keras.models.load_model('keras/policy_network_red_vel_x')
policy_network_red_vel_y = tf.keras.models.load_model('keras/policy_network_red_vel_y')
policy_network_red_vel_a = tf.keras.models.load_model('keras/policy_network_red_vel_a')

with tf.device('/GPU:0'):
    while not done:
        steps += 1
        action_holder_converted = {}

        for agent in ["blue_1", "blue_2", "blue_3", "red_1", "red_2", "red_3"]:
            action_probs_vel_x, action_probs_vel_y, action_probs_vel_a = 0, 0, 0
            if agent.startswith("red"):
            # Get the action probabilities from the policy network
                action_probs_vel_x = policy_network_red_vel_x.predict(np.array([state[agent]]))[0]
                action_probs_vel_y = policy_network_red_vel_y.predict(np.array([state[agent]]))[0]
                action_probs_vel_a = policy_network_red_vel_a.predict(np.array([state[agent]]))[0]
            else:
                action_probs_vel_x = policy_network_blue_vel_x.predict(np.array([state[agent]]))[0]
                action_probs_vel_y = policy_network_blue_vel_y.predict(np.array([state[agent]]))[0]
                action_probs_vel_a = policy_network_blue_vel_a.predict(np.array([state[agent]]))[0]

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

            action_holder_converted[agent] = holder

        next_state, reward, dones, truncation, info = environment.step(action_holder_converted)  # take a step in the environment
        if truncation['__all__']:
            done = True
        environment.reset_pygame()
        environment.render()

        # convert image to pygame surface object

        count += 1


        state = next_state


