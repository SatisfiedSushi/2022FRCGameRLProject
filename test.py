from gymnasium.spaces import Dict, Box, MultiDiscrete, Discrete
import numpy as np
import pprint

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
