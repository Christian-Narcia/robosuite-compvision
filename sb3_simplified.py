# import gymnasium as gym
import os
os.environ['MUJOCO_GL'] = 'glfw'  # Set the rendering backend to 'glfw'
import numpy as np
import robosuite as suite
import time
from robosuite.wrappers.gym_wrapper import GymWrapper
import torch
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


######################### testing camera setup

# env = suite.make(      
#     env_name=envName,
#     robots="UR5ev2",
#     has_renderer=True,
#     has_offscreen_renderer=True,
#     use_camera_obs=True,
#     camera_names=["topdown"],
#     camera_heights=64,
#     camera_widths=64,
#     use_object_obs=False,
#     reward_shaping=False
# )
# env = GymWrapper(env)

# env.reset()
# obs = env.reset()
# print("obs", obs)

# img_dim = 64
# # Known dimensions
# image_dim = img_dim * img_dim * 3  # for 64x64 12,288, for 128x128 196,608

# # Split observations
# camview_image = obs[:, :image_dim]
# robot0_proprio_state = obs[:, image_dim:]
# print(len(camview_image))
# print(len(robot0_proprio_state))
# exit()

# print("obs.keys()", obs.keys())
# # env.viewer.set_camera(camera_id=0)

# # Get action limits
# low, high = env.action_spec

# action = np.random.uniform(low, high)
# obs, reward, done, _ = env.step(action)
# # print(len(obs[0]))


# plt.imshow(obs["topdown_image"], origin='lower')
# plt.axis('off')
# plt.title("Top down Table View Image")
# plt.show()


# # do visualization
# for i in range(10000):
#     action = np.random.uniform(low, high)
#     obs, reward, done, _ = env.step(action)
#     env.render()



# set KMP_DUPLICATE_LIB_OK=TRUE


env = suite.make(      
    env_name=envName,
    robots="UR5ev2",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    # camera_names=["topdown"],
    # use_object_obs=False,
    reward_shaping=True
)
vec_env = GymWrapper(env)
# vec_env.reset()
# obs = vec_env.reset()
# print(len(obs[0]))
# print(obs)

version = "v5"
# exit()
device ='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./runs/PPO_"+envName+"_sb3_simplified"+version+"/",device = device)
model.learn(total_timesteps=1000000)

model.save("./runs/PPO_"+envName+"_sb3_simplified"+version+"/"+envName+"_ppo_reach_simplified"+version)
# model.save("./runs/PPO_"+envName+"_sb3_simplifiedv2_No_Object_Obs/"+envName+"_ppo_reach_simplifiedv2_No_Object_Obs")