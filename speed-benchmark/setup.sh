# Precheck
echo "First checking if GPU detected..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.__version__)'

echo "Press Enter to continue..."
read -r

# https://cloud.vast.ai/
# https://www.runpod.io/console/pods
# Runpod or not
# RUNPOD_POD_HOSTNAME

# System Packages
apt update
apt install -y curl wget git
apt install -y dstat htop inxi nvtop
apt install -y byobu neovim

# Always use /workspace (easier)
mkdir -p /workspace

# Install Mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p /workspace/miniforge3

# Setup
/workspace/miniforge3/bin/mamba init

# Force bashrc to run in interactive mode
PS1='$ '
source ~/.bashrc
unset PS1

# Install requirements
mamba install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba upgrade ffmpeg -y

pip install transformers datasets accelerate
pip install loguru
pip install git+https://github.com/m-bain/whisperx.git
pip install faster-whisper -U
pip install librosa soundfile -U

export LD_LIBRARY_PATH=/workspace/miniforge3/lib/python3.10/site-packages/torch/lib
