# System
apt install htop inxi nvtop
apt install neovim

# Install Mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Setup
~/miniforge3/bin/mamba init
source ~/.bashrc

# Install requirements
mamba install -c "nvidia/label/cuda-12.1.1" cuda-toolkit -y
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
# mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba update ffmpeg

pip install transformers dataset accelerate
pip install loguru
pip install git+https://github.com/m-bain/whisperx.git
pip install faster-whisper -U
pip install librosa soundfile -U

export LD_LIBRARY_PATH=/root/miniforge3/lib/python3.10/site-packages/torch/lib
