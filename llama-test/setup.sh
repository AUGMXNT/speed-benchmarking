# vast images require this
apt install -y build-essential

# vast images have an LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/workspace/miniforge3/lib
export MAKEFLAGS="-j$(nproc)"

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=on
cmake --build . --config Release
# make -j$(nproc) LLAMA_CUBLAS=1
