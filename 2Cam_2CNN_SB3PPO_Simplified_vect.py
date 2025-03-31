import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


################################################################
#               Custom Callback for CNN Heatmaps
################################################################

class ImageLoggingCallback(BaseCallback):
    """
    A custom callback that, at intervals, logs CNN "heatmap" overlays
    for both front-view (channels 0..2) and side-view (channels 3..5) images
    directly to TensorBoard.
    """
    def __init__(
        self,
        log_dir: str = "./logs/tensorboard",
        log_freq: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        """
        Called each time we call `env.step()`.
        Only log if the total number of timesteps so far is divisible by log_freq.
        """
        if self.model.num_timesteps % self.log_freq == 0:
            if self.verbose:
                print(f"[ImageLoggingCallback] Logging heatmaps at step {self.model.num_timesteps}")
            # Sample some observations:
            obs = self._sample_observations()
            # Generate and log heatmaps:
            self._log_cnn_heatmaps(obs)
        return True

    def _sample_observations(self):
        """
        Grab observations from the environment by doing a reset().
        For a VecEnv, shape will be (n_envs, obs_dim).
        We'll just use the first environment's data for visualization.
        """
        obs = self.training_env.reset()
        return obs

    def _log_cnn_heatmaps(self, obs):
        """
        1) Convert the first environment's observation to Torch.
        2) Pass it through the CNN to get intermediate feature maps.
        3) For each layer's feature map, compute average activation (H,W).
        4) Overlay that heatmap on both front-view & side-view images.
        5) Log those overlays to TensorBoard.
        """
        # Just the first env obs
        obs_single = obs[0]  # shape: (obs_dim,)
        obs_single = torch.tensor(obs_single, dtype=torch.float).unsqueeze(0).to(self.model.device)

        # Slice out the image portion: 3 × 64 × 64 = 24,576
        image_dim = numLayers * img_dim * img_dim  # 3 * 64 * 64 = 24576
        image_obs1 = obs_single[:, :image_dim]      # shape = (1, 24576)
        image_obs2 = obs_single[:, image_dim:2*image_dim] 

        # Reshape to (batch=1, channels=6, H=64, W=64)
        image_obs1 = image_obs1.view(-1, numLayers, img_dim, img_dim)
        image_obs2 = image_obs2.view(-1, numLayers, img_dim, img_dim)

        # Forward pass through the custom CNN
        cnn_module1 = self.model.policy.features_extractor.cnn1
        cnn_module2 = self.model.policy.features_extractor.cnn2
        with torch.no_grad():
            # cnn_module returns (cnn_output, [feat1, feat2, feat3])
            # each featN has shape (batch, channels, H, W)
            _, feature_maps1 = cnn_module1(image_obs1)
            _, feature_maps2 = cnn_module2(image_obs2)

        # The input image as float in [0,1].
        # front-view = channels [0..2], side-view = [3..5].
        # frontview_img = (image_obs1 / 255.0).cpu()  # shape (3,64,64)
        # sideview_img  = (image_obs2 / 255.0).cpu()  # shape (3,64,64)
        frontview_img = (image_obs1 / 255.0).cpu().squeeze(0)  # Now shape is (3,64,64)
        sideview_img  = (image_obs2 / 255.0).cpu().squeeze(0)  # Same shape

        # For each layer's feature map, create & log an overlay:
        for i, (f_map1, f_map2) in enumerate(zip(feature_maps1, feature_maps2)):
            # f_map1 and f_map2 have shape (1, channels, H, W) => pick batch=0
            f_map1_single = f_map1[0]  # shape (channels, H, W)
            f_map2_single = f_map2[0]  # shape (channels, H, W)

            # 1) average across channels => shape (H, W)
            heatmap1 = f_map1_single.mean(dim=0).cpu().numpy()  # Front view heatmap
            heatmap2 = f_map2_single.mean(dim=0).cpu().numpy()  # Side view heatmap

            # Normalize to [0,1]
            heatmap1 -= heatmap1.min()
            heatmap1 /= (heatmap1.max() + 1e-8)

            heatmap2 -= heatmap2.min()
            heatmap2 /= (heatmap2.max() + 1e-8)

            # 2) Overlay on front-view image (using feature_maps1)
            fig_front = self._overlay_heatmap(frontview_img, heatmap1)
            front_tensor = self._figure_to_image_tensor(fig_front)
            self.writer.add_image(
                f"heatmap/layer_{i}/topdown",
                front_tensor,
                global_step=self.model.num_timesteps
            )
            plt.close(fig_front)

            # 3) Overlay on side-view image (using feature_maps2)
            fig_side = self._overlay_heatmap(sideview_img, heatmap2)
            side_tensor = self._figure_to_image_tensor(fig_side)
            self.writer.add_image(
                f"heatmap/layer_{i}/side",
                side_tensor,
                global_step=self.model.num_timesteps
            )
            plt.close(fig_side)


    def _overlay_heatmap(self, img_3ch, heatmap):
        """
        Create a single matplotlib figure with `img_3ch` in the background
        and a semi-transparent heatmap on top.
        img_3ch: shape (3, H, W), in [0,1]
        heatmap: shape (H, W),    in [0,1]
        Returns a Matplotlib Figure object.
        """
        # Convert background to shape (H, W, 3)
        background = img_3ch.permute(1, 2, 0).numpy()  # (64,64,3)

        fig = plt.figure(figsize=(0.64, 0.64))
        plt.imshow(background)  # show the raw image
        plt.imshow(heatmap, alpha=0.3, cmap='jet')  # overlay heatmap
        plt.axis('off')
        return fig


    def _figure_to_image_tensor(self, fig):

        fig.canvas.draw()

        # get_width_height() returns (width, height)
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

        # Now reshape as (height, width, 3), matching the order
        # buf = buf.reshape(height, width, 3)
        buf = buf.reshape(height, width, 4)
        buf = buf[:, :, 1:]  # Drop the alpha channel, keep R, G, B


        # Convert to torch tensor with shape (3, height, width)
        image_tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
        return image_tensor

    def _on_training_end(self) -> None:
        """
        Close the SummaryWriter.
        """
        self.writer.close()


################################################################
#        Custom CNN Feature Extractor for 3-Channel Input
################################################################

numLayers = 3
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
        # x shape: (batch, 3, 64, 64)
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
    Slices out the image portion, reshapes to (N,3,64,64),
    then passes through CNN. Optionally can include proprio obs.
    """
    def __init__(self, observation_space, features_dim=img_dim, use_proprio_obs=False):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.cnn1 = CNNFeatures()
        self.cnn2 = CNNFeatures()
        self.use_proprio_obs = use_proprio_obs
        
        # Check CNN output dimension
        with torch.no_grad():
            sample_image = torch.zeros(1, numLayers, img_dim, img_dim)
            cnn1_output, _ = self.cnn1(sample_image)
            cnn2_output, _ = self.cnn2(sample_image)
            cnn1_output_dim = cnn1_output.shape[1]
            cnn2_output_dim = cnn2_output.shape[1]
        
        # If we were using proprio, specify dimension
        if self.use_proprio_obs:
            self.proprio_dim = 41  # environment obs number
        else:
            self.proprio_dim = 0
        
        self._features_dim = cnn1_output_dim + cnn2_output_dim + self.proprio_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Number of pixels in the image portion
        image_dim = numLayers * img_dim * img_dim  # 3*64*64=12,288

        # Split observation
        cam1view_image = observations[:, :image_dim]
        cam2view_image = observations[:, image_dim:image_dim*2]
        robot0_proprio_state = observations[:, image_dim*2:]  # if any

        # Reshape image => (batch,3,64,64)
        cam1view_image = cam1view_image.view(batch_size, numLayers, img_dim, img_dim)
        cam2view_image = cam2view_image.view(batch_size, numLayers, img_dim, img_dim)
        # Normalize from [0,255] => [0,1]
        cam1view_image = cam1view_image / 255.0
        cam2view_image = cam2view_image / 255.0

        # Pass image through CNN
        cnn1_output, _ = self.cnn1(cam1view_image)
        cnn2_output, _ = self.cnn2(cam2view_image)
        # print(len(cnn_output[0]))

        # If using proprio, concat
        if self.use_proprio_obs:
            features = torch.cat([cnn1_output, cnn2_output, robot0_proprio_state], dim=1)
        else:
            features = torch.cat([cnn1_output, cnn2_output], dim=1)

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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    def _init():
        env = suite.make(
            env_name=envName,
            camera_names=["topdown", "sideview"],
            robots="UR5ev2",
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=False,  # no object info
            camera_heights=img_dim,
            camera_widths=img_dim,
            reward_shaping=True,
        )
        env = GymWrapper(env)
        env = Monitor(env)  # Add this line
        return env
    return _init
################################################################
#               Main Training Code
################################################################
if __name__ == "__main__":
    # Basic parameters
    envName = "reachImgSimplified"
    version = "v6"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    pathDir = "../runs/"


    # Generate a unique run name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_{timestamp}"

    base_log_dir = f"{pathDir}2Img2CNNPPO_{envName}_sb3_simplified_vect_{version}"
    full_log_dir = os.path.join(base_log_dir, run_name)
    print(full_log_dir)
    # Create the folder ahead of time (optional, SB3 will also create it)
    os.makedirs(full_log_dir, exist_ok=True)
    # exit()

    env_config = {
    "Reach": 200,  # epLen for Reach environment
    "Lift": 1000,    # epLen for Lift environment
    "reachImgSimplified" : 200
    }

    num_envs = 1  # Number of parallel environments
    epLen = env_config[envName]
    n_steps = epLen * num_envs
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])



    # Instantiate custom policy with PPO
    # Now pass this to SB3 and your image logger
    model = PPO(
        policy=CustomActorCriticPolicy,
        env=vec_env,
        verbose=1,
        tensorboard_log=full_log_dir,  # this is still the parent
        device=device
    )

    image_logging_callback = ImageLoggingCallback(
        log_dir=os.path.join(full_log_dir, "images"),  # your own custom log path
        log_freq=10000,
        verbose=1
    )

    # Train
    model.learn(
        total_timesteps=1000000,
        callback=image_logging_callback
    )

    # Save final model
    os.makedirs(f"{pathDir}Img2CNNPPO_{envName}_sb3_simplified_{version}/", exist_ok=True)
    model.save(f"{pathDir}Img2CNNPPO_{envName}_sb3_simplified_{version}/{envName}_Img_2CNN_ppo_reach_simplified_{version}")
    print("Training complete and model saved.")
