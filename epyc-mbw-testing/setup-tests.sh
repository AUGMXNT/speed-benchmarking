#!/bin/bash

echo "---"
echo "Setting up sysbench"
sudo pacman -S sysbench

echo "---"
echo "Setting up Likwid"
git clone https://github.com/RRZE-HPC/likwid
cd likwid 
sudo make install
cd ..

echo "---"
echo "Setting up stream"
git clone https://github.com/arstgr/stream
cd stream
sh build_stream.sh
cd ..

echo "---"
echo "Setting up pbmw"
git clone https://github.com/bingmann/pmbw
cd pbmw
make
cd ..

echo "---"
echo "Setting up llama.cpp"
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j -v
cd ..
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
