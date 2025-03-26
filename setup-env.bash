# !/usr/bin/env bash

# To run this bash script, it is assumed you have anaconda activatated and have already downloaded the downloaded robosuite-compvision repo below and are inside the folder.
# It is also assumed you have anaconda installed with (base) env in the terminal
# If you are running on the cluster, you need to change the torchvision version or you will have an error
# github repo https://github.com/Christian-Narcia/robosuite-compvision.git

# Define environment name and requirements file
ENV_NAME="robo4"
REPO_URL="https://github.com/Christian-Narcia/robosuite-compvision.git"
REQUIREMENTS_FILE="requirements.txt"

# Create a new conda environment
conda create --name $ENV_NAME python=3.10 -y || { echo "Failed to create $ENV_NAME"; exit 1; }

# # Activate the conda environment
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

# conda install git -y
# # Clone the repository
# git clone $REPO_URL
REPO_NAME=$(basename "$REPO_URL" .git)

# Change to the cloned repository directory
# cd $REPO_NAME || { echo "Failed to change directory to $REPO_NAME"; exit 1; }

# Install dependencies using pip
pip install -r $REQUIREMENTS_FILE

pip install torch==2.6
pip install torchvision==0.21
pip install tensorboard
pip install stable_baselines3
pip install h6py


echo "Setup complete. The environment '$ENV_NAME' is ready."