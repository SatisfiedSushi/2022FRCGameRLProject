import math

from gymnasium.spaces import Dict, Box, MultiDiscrete, Discrete
import numpy as np
import tensorflow as tf
import pprint
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

observations = Dict(
    {
        'velocity': Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,)),
        'angular_velocity': Box(low=np.array([-1]), high=np.array([1]), shape=(1,)),
        'angle': Box(low=np.array([-1]), high=np.array([360]), shape=(1,)),
        'closest_ball': Dict(
            {
                'angle': Box(low=np.array([-180]), high=np.array([180]), shape=(1,)),
            }
        ),
        'robots_in_sight': Dict(
            {
                'angles': Box(low=np.array([-180, -180, -180, -180, -180]), high=np.array([180, 180, 180, 180, 180]), shape=(5,)),
                'teams': MultiDiscrete(np.array([3, 3, 3, 3, 3])),
            }
        ),
    }
)

#convert everything above to only one box
observations = Box(low=np.array([-1, -1, -1, -1, -180, -180, -180, -180, -180, -180, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 360, 180, 180, 180, 180, 180, 180, 3, 3, 3, 3, 3]), shape=(15,))

actions = Dict(
    {
        'velocity': Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,)),
        'angular_velocity': Box(low=np.array([-1]), high=np.array([1]), shape=(1,)),
    }
)
'''
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(observations.sample())'''
print(f'keras/{str(["policy_network_blue_vel_x", "policy_network_blue_vel_y", "policy_network_blue_vel_a"][[9, 8, 7].index(8)])}')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(np.random.choice(5, 1, replace=False, p=[0.1, 0, 0.3, 0.6, 0]))
def convert_LL_angle_rewards(angle):
    LL_FOV = 31.65
    multiplied_angle = angle * 100

    if angle == -1:
        return 0
    elif angle < 0.01:
        return LL_FOV
    else:
        return ((LL_FOV * 100) - multiplied_angle)/100


def convert_LL_distance_rewards(distance):
    max_y_distance = 8.23
    max_X_distance = 16.46
    max_distance = math.sqrt(max_y_distance ** 2 + max_X_distance ** 2)
    multiplied_distance = distance * 100
    if distance == -1:
        return 0
    if distance < 0.01:
        return max_distance
    else:
        return ((max_distance * 100) - multiplied_distance)/100
print(convert_LL_angle_rewards(0.01))
print(convert_LL_distance_rewards(0.01))






