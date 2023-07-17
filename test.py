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

actions = Dict(
    {
        'velocity': Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,)),
        'angular_velocity': Box(low=np.array([-1]), high=np.array([1]), shape=(1,)),
    }
)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(observations.sample())
