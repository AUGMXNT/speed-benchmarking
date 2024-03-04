git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc) LLAMA_CUBLAS=1
