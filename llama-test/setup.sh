# vast images require this
apt install -y build-essential

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc) LLAMA_CUBLAS=1
