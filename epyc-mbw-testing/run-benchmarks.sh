#!/usr/bin/env bash
#
# Usage: ./run-benchmarks.sh <run-name>
#
# This will create a directory named <run-name> and place output logs there.

set -euo pipefail

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo or switch to the root user." >&2
    exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run-name>"
  exit 1
fi

RUN_NAME="results-$1"
mkdir -p "$RUN_NAME"

########################################
# System Setup & Info Collection
########################################
echo "=== Disabling automatic NUMA balancing ==="
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

echo "=== Saving system info to ${RUN_NAME}/system-info.txt ==="
{
  echo "=== Benchmark run: $RUN_NAME ==="
  date

  echo
  echo "=== uname -a ==="
  uname -a

  echo
  echo "=== lscpu ==="
  lscpu

  echo
  echo "=== numactl --hardware ==="
  numactl --hardware || true

  echo
  echo "=== dmidecode ==="
  numactl --hardware || true
} > "${RUN_NAME}/system-info.txt"


########################################
# likwid-bench: Memory Tests (~5min)
########################################
# We'll test scalar & AVX-512 versions of copy_mem, stream_mem, triad_mem
BENCHMARKS=(
  "copy_mem"
  "copy_mem_avx512"
  "stream_mem"
  "stream_mem_avx512"
  "triad"
  "triad_avx512"
  "triad_mem_avx512"
)

# Adjust to match your system
MEM_SIZE="256GB"
NUM_THREADS="48"  # or however many logical cores/hw threads you want to use

echo "=== Running likwid-bench tests ==="
for bm in "${BENCHMARKS[@]}"; do
    echo
    echo "-------------------------------------------------"
    echo "Dropping caches and running $bm ..."
    echo "-------------------------------------------------"
    # Drop caches each time to get a consistent "cold" memory start
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

    # Use 'time' and 'tee' to get both timing and output
    time nice -n -20 numactl --interleave=all \
    likwid-bench -t "${bm}" -w N:"${MEM_SIZE}":"${NUM_THREADS}" 2>&1 | tee -a "${RUN_NAME}/likwid-bench-${bm}.log"
done


########################################
# PMBW (~45min)
########################################
# If you also want to run pmbw each time:
if [ -x "./pmbw/pmbw" ]; then
  echo
  echo "=== Running pmbw ==="
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  # Example pmbw command
  time nice -n -20 numactl --interleave=all \
    ./pmbw/pmbw -f PermRead64UnrollLoop -o "${RUN_NAME}/pmbw-stats.txt" -s 1024 -S 4294967296 -Q -p 1 -P 48 2>&1 | tee -a "${RUN_NAME}/pmbw.log"
else
  echo "pmbw binary not found; skipping."
fi


########################################
# AOCC STREAM (~2min)
########################################
# If you compiled the AOCC STREAM to `./stream/stream`, run it here:
if [ -x "./stream/stream" ]; then
  echo
  echo "=== Running AOCC STREAM ==="
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  time nice -n -20 numactl --interleave=all \
    ./stream/stream 2>&1 tee -a "${RUN_NAME}/aocc-stream.log"
else
  echo "AOCC STREAM binary not found; skipping."
fi


########################################
# sysbench (~30s)
########################################
if which sysbench > /dev/null 2>&1; then
  echo
  echo "=== Running sysbench (avg) ==="
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  time nice -n -20 numactl --interleave=all \
    ./sysbench-memory-read-avg.sh | tee -a "${RUN_NAME}/sysbench-avg.log"
else
  echo "sysbench not found; skipping."
fi

########################################
# llama-bench (~1min)
########################################
if [ -x "./llama.cpp/build/bin/llama-bench" ]; then
  echo
  echo "=== Running llama-bench ==="
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  time nice -n -20 numactl --interleave=all \
   ./llama.cpp/build/bin/llama-bench -r 10 --numa numactl -fa 1 -m llama-2-7b.Q4_0.gguf -o json | tee -a "${RUN_NAME}/llama-bench.json"
else
  echo "llama-bench not found; skipping."
fi


########################################
# 6. Done
########################################

# if sudo'd let's change perms for results
ORIGINAL_USER=${SUDO_USER:-root}  # Default to root if SUDO_USER is not set
chown -R "$ORIGINAL_USER":"$ORIGINAL_USER" "$RUN_NAME"

echo
echo "=== All benchmarks completed. Logs are in '${RUN_NAME}' directory. ==="

