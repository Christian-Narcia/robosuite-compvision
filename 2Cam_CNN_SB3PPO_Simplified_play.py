import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robosuite.wrappers import GymWrapper
import robosuite as suite

import os

from stable_baselines3.common.callbacks import BaseCallback
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

envName = "reachImgSimplified"
img_dim = 64
use_proprio_obs = False 
# # Create environment instance
# env = suite.make(
#     # env_name="Reach",
#     # camera_names="frontview",
#     # robots="UR5e",
#     env_name=envName,
#     camera_names=["topdown"],
#     robots="UR5ev2",
#     has_offscreen_renderer=True,
#     use_camera_obs=True,
#     use_object_obs=False,  # Exclude object observations
#     camera_heights=img_dim,
#     camera_widths=img_dim,
#     reward_shaping=True
# )

# # Wrap the environment
# vec_env = GymWrapper(env)

def make_env(img_dim=64, use_proprio_obs=False):
    """
    Utility function to recreate the same environment used in training.
    """
    envName = "reachImgSimplified"  # must match your training environment name
    env = suite.make(
        env_name=envName,
        camera_names=["topdown", "sideview"],  # must match your training
        robots="UR5ev2",
        has_offscreen_renderer=True,
        # render_camera="sideview",
        has_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_heights=img_dim,
        camera_widths=img_dim,
        reward_shaping=True
    )
    # Wrap with GymWrapper
    return GymWrapper(env)


################################################################
#        Custom CNN Feature Extractor for 6-Channel Input
################################################################

numLayers = 6
img_dim = 64

class CNNFeatures(nn.Module):
    """
    Simple 3-layer CNN that also returns intermediate feature maps.
    """
    def __init__(self):
        super(CNNFeatures, self).__init__()
        self.conv1 = nn.Conv2d(numLayers, 16, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

    def forward(self, x):
        # x shape: (batch, 6, 64, 64)
        x = self.conv1(x)
        x = self.relu1(x)
        feat1 = x.clone()

        x = self.conv2(x)
        x = self.relu2(x)
        feat2 = x.clone()

        x = self.conv3(x)
        x = self.relu3(x)
        feat3 = x.clone()

        x = self.flatten(x)
        # Return final features + a list of intermediate feats
        return x, [feat1, feat2, feat3]


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Slices out the image portion, reshapes to (N,6,64,64),
    then passes through CNN. Optionally can include proprio obs.
    """
    def __init__(self, observation_space, features_dim=img_dim, use_proprio_obs=False):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.cnn = CNNFeatures()
        self.use_proprio_obs = use_proprio_obs
        
        # Check CNN output dimension
        with torch.no_grad():
            sample_image = torch.zeros(1, numLayers, img_dim, img_dim)
            cnn_output, _ = self.cnn(sample_image)
            cnn_output_dim = cnn_output.shape[1]
        
        # If we were using proprio, specify dimension
        if self.use_proprio_obs:
            self.proprio_dim = 41  # or whatever dimension your environment has
        else:
            self.proprio_dim = 0
        
        self._features_dim = cnn_output_dim + self.proprio_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Number of pixels in the image portion
        image_dim = numLayers * img_dim * img_dim  # 6*64*64=24576

        # Split observation
        camview_image = observations[:, :image_dim]
        robot0_proprio_state = observations[:, image_dim:]  # if any

        # Reshape image => (batch,6,64,64)
        camview_image = camview_image.view(batch_size, numLayers, img_dim, img_dim)
        # Normalize from [0,255] => [0,1]
        camview_image = camview_image / 255.0

        # Pass image through CNN
        cnn_output, _ = self.cnn(camview_image)
        print(len(cnn_output))

        # If using proprio, concat
        if self.use_proprio_obs:
            features = torch.cat([cnn_output, robot0_proprio_state], dim=1)
        else:
            features = cnn_output

        return features


################################################################
#             Custom Actor-Critic Policy
################################################################

from stable_baselines3.common.policies import ActorCriticPolicy

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs={'features_dim': img_dim, 'use_proprio_obs': False},
            **kwargs
        )




# device ='cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# # Instantiate the PPO model with the custom policy
# model = PPO(
#     policy=CustomActorCriticPolicy,
#     env=vec_env,
#     verbose=1,
#     tensorboard_log="./runs/ImgPPO_"+envName+"_sb3/",
#     device = device
# )

# Train the model
# model.learn(total_timesteps=2000000)

# # model.save("./runs/ImgPPO_"+envName+"_sb3/"+envName+"_Img_ppo_reach_simplified_1")

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model.save(f"./runs/ImgPPO_{envName}_sb3/{envName}_Img_ppo_reach_simplified_1_{timestamp}")


# # Instantiate your callback
# image_logging_callback = ImageLoggingCallback(
#     log_dir="./runs/ImgPPO_"+envName+"_sb3/tf_logs",
#     log_freq=10000, 
#     verbose=1
# )

# # Train the model with the callback
# model.learn(
#     total_timesteps=200000, 
#     callback=image_logging_callback
# )


# model.save("./runs/ImgPPO_"+envName+"_sb3/"+envName+"_Img_ppo_reach_simplified_1")

def run_playback(model_path, n_steps=1000, render=False):
    """
    Load the trained model and run a simple rollout.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1) Recreate the environment
    env = make_env(img_dim=64, use_proprio_obs=False)

    # 2) Load model
    # NOTE: We must ensure the custom policy class is known. One way is via custom_objects:
    model = PPO.load(
        model_path,
        device=device,
        # If needed, you can specify custom_objects={"policy_class": CustomActorCriticPolicy}
        # However, if the class name inside the loaded model is identical to
        # "CustomActorCriticPolicy" in the same scope, SB3 usually picks it up automatically.
    )

    obs, _ = env.reset()
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info, _ = env.step(action)

        # If you want some textual logging:
        # print(f"Step={step}, Reward={reward}, Done={done}")

        # Robosuite environment default rendering:
        env.render()  # Make sure your environment's renderer is enabled

        if done:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    # Example usage
    # Update the path to your trained model zip
    # model_path = "./runs/ImgPPO_reachImgSimplified_sb3_simplified_v5/reachImgSimplified_Img_ppo_reach_simplified_v5.zip"
    # model_path = "./runs/ImgPPO_reachImgSimplified_sb3_simplified_v4/reachImgSimplified_Img_ppo_reach_simplified_v4.zip"
    model_path = "../cluster-robo-04_12/runs/ImgPPO_reachImgSimplified_sb3_simplified_v5/reachImgSimplified_Img_ppo_reach_simplified_v5.zip"



    
    run_playback(model_path, n_steps=2000, render=False)
