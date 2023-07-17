import random
import time

from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from typing import Dict
# Example: using a multi-agent env
env = MultiAgentCartPole({"num_agents": 3, "render_mode": "human"})

# Observations are a dict mapping agent names to their obs. Not all
# agents need to be present in the dict in each time step.
print(env.reset())
# Actions should be provided for each agent that returned an observation.
new_obs, rewards, dones, truncated, infos = env.step(action_dict={"0": 1, "1": 1, "2": 1})

# Similarly, new_obs, rewards, dones, infos, etc. also become dicts
print(rewards)

# Individual agents can early exit; env is done when "__all__" = True
print(dones)

def is_all_done(done: Dict) -> bool:
    for key, val in done.items():
        if not val:
            return False
    return True


while not is_all_done(dones):
    action_dict = {"0": random.randint(0, 1), "1": random.randint(0, 1), "2": random.randint(0, 1)}

    new_obs, rewards, dones, truncated, infos = env.step(action_dict)
    print("Reward: ", rewards)
    time.sleep(.1)
    env.render()