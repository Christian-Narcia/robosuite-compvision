# import gymnasium as gym
import os
os.environ['MUJOCO_GL'] = 'glfw'  # Set the rendering backend to 'glfw'
import numpy as np
import robosuite as suite
import time
from robosuite.wrappers.gym_wrapper import GymWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv

import matplotlib
matplotlib.use('TkAgg')  # Change the backend to avoid Qt conflicts
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3 import PPO

# envName = "Lift"
envName = "reachImgSimplified"


# def make_env():
#     def _init():
#         env = suite.make(
#             # env_name="Reach",
#             env_name=envName,
#             robots="UR5e",
#             has_offscreen_renderer=False,
#             use_camera_obs=False,
#             # use_object_obs=False,
#             reward_shaping=False
#         )
#         env = GymWrapper(env)
#         env = Monitor(env)  # Add this line
#         return env
#     return _init

# if __name__ == '__main__':
#     tSteps = 4000000
#     num_envs = 4
#     epLen = 1000
#     n_steps = epLen * 4
#     vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])


#     env.reset()
#     env.viewer.set_camera(camera_id=0)

#     # Get action limits
#     low, high = env.action_spec

#     # do visualization
#     for i in range(10000):
#         action = np.random.uniform(low, high)
#         obs, reward, done, _ = env.step(action)
#         env.render()


    # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./runs_simplified/PPO_"+envName+"_vec_sb3/", n_steps = n_steps, device = 'cuda')
    # model.learn(total_timesteps=tSteps)
    # model.save("runs_simplified/ppo_"+envName+"2m_vec_sparse_1")


img_dim = 64

env = suite.make(      
    env_name=envName,
    # robots="UR5e",
    robots="UR5ev2",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    # camera_names=["topdown"],
    camera_names=["frontview","sideview"],
    render_camera="birdview",
    camera_heights=img_dim,
    camera_widths=img_dim,
    # use_object_obs=False,
    reward_shaping=False
)
# env = GymWrapper(env)

env.reset()
obs = env.reset()

print("obs.keys()", obs.keys())
# print(obs["robot0_proprio-state"])
# env.viewer.set_camera(camera_id=0)

# Get action limits
low, high = env.action_spec
# print("low", low)
# print("high", high)
action = np.random.uniform(low, high)
# action = [-1.,-1.,-1.,-1.,-1.,-1.,-1.]
# action = [0.,0.,0.,0.,0.,0.,0.]
# print("action", action)
obs, reward, done, _= env.step(action)
# # print(len(obs[0]))

# plt.imshow(obs["topdown_image"], origin='lower')
# plt.axis('off')
# plt.title("Top Down Image")
# plt.show()

# plt.imshow(obs["sideview_image"], origin='lower')
# plt.axis('off')
# plt.title("Side View Image")
# plt.show()

# exit()

# # # do visualization
for i in range(10000):
    action = np.random.uniform(low, high)
    # action = [-1.,-1.,-1.,-1.,-1.,-1.,-1.]
    # action = [0.,0.,0.,0.,0.,0.,-1.]
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(1/30)
