#!/bin/bash

# Get current date and time
current_date=$(date)

# Get distro information
if [ -f /etc/os-release ]; then
    distro=$(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '"')
else
    distro="Unknown"
fi

# Get kernel version
kernel=$(uname -r)

# Get CPU information
cpu=$(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)
sockets=$(lscpu | grep 'Socket(s)' | cut -d':' -f2 | xargs)
cores_per_socket=$(lscpu | grep 'Core(s) per socket' | cut -d':' -f2 | xargs)
threads_per_core=$(lscpu | grep 'Thread(s) per core' | cut -d':' -f2 | xargs)
cpu_max_mhz=$(lscpu | grep 'CPU max MHz' | cut -d':' -f2 | xargs)
cpu_max_mhz=$(awk "BEGIN {printf \"%.1f GHz\", $cpu_max_mhz / 1024}")
cores=$((sockets * cores_per_socket))
threads=$((cores * threads_per_core))

# Get RAM information
total_ram=$(free | awk '/^Mem:/{printf "%.1f GiB", $2/1024/1024}')

# memory_gib=$(awk "BEGIN {printf \"%.1f\", $memory_mib / 1024}")


# Check for Nvidia GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia_gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
        name=$(echo "$line" | cut -d',' -f1)
        memory=$(echo "$line" | cut -d',' -f2)
	memory_mib=$(echo $memory | awk '{print $1}')
        memory_gib=$(awk "BEGIN {printf \"%.1f\", $memory_mib / 1024}")
        echo "  $name ($memory_gib GiB)"
    done)
else
    nvidia_gpus=""
fi

# Check for AMD GPU using rocminfo
if command -v rocminfo &> /dev/null; then
    amd_gpus=$(rocminfo | awk '
        /Device Type/ {
            dev_type = $3
            vram = ""
            in_pool_1 = 0
        }
        /Marketing Name/ {
            marketing = $3 " " $4 " " $5 " " $6
        }
        /Pool 1/ {
            in_pool_1 = 1
        }
        in_pool_1 && /Size:/ {
            match($0, /Size: +([0-9]+)/, arr)
            if (arr[1] != "") {
                vram_gib = arr[1] / 1024 / 1024
                vram = sprintf("%.1f GiB", vram_gib)
            }
            in_pool_1 = 0  # reset after finding the size in Pool 1
        }
        dev_type == "GPU" && marketing != "" && vram != "" {
            print "  " marketing " (" vram ")"
            dev_type = ""
            marketing = ""
            vram = ""
        }
    ')
else
    amd_gpus=""
fi


# Check for CUDA
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep 'release' | awk '{print $5,$6}')
else
    cuda_version=""
fi

# Check for ROCm
if command -v hipcc &> /dev/null; then
    rocm_version=$(hipcc --version | grep 'HIP version' | awk '{print $3}')
else
    rocm_version=""
fi

# Check if torch is installed and print the version
if python -c "import torch" &> /dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
else
    torch_version=""
fi

# Print the system information
echo "$current_date"
echo "==="
echo "OS: $distro ($kernel)"
echo "CPU: $cpu (${cores}C ${threads}T @ $cpu_max_mhz)"
echo "RAM: $total_ram"

if [ -n "$nvidia_gpus" ]; then
    echo "Nvidia GPUs:"
    echo "$nvidia_gpus"
fi
if [ -n "$cuda_version" ]; then
    echo "CUDA Version: $cuda_version"
fi

if [ -n "$amd_gpus" ]; then
    echo "AMD GPUs:"
    echo "$amd_gpus"
fi

if [ -n "$rocm_version" ]; then
    echo "ROCm Version: $rocm_version"
fi

if [ -n "$torch_version" ]; then
    echo "PyTorch: $torch_version"
fi
