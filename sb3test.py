import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env# Parallel environments
from FRCGameEnvSingle import env

vec_env = make_vec_env(env_id=env, n_envs=1)

# model = PPO("MlpPolicy", vec_env, verbose=1)
model = PPO.load("E:\SwerveSim\ppo_cartpole", vec_env)
model.learn(total_timesteps=50000000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

'''model = PPO.load("E:\SwerveSim\ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")'''