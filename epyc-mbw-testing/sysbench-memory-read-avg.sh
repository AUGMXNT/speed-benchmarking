#!/usr/bin/env bash

# Usage:
#   ./test_mem_throughput.sh [runs]
# Example:
#   ./test_mem_throughput.sh 5

# Number of times to run sysbench (default is 20).
NUM_RUNS="${1:-20}"

TOTAL=0

for i in $(seq 1 "${NUM_RUNS}"); do
  # Run sysbench with nice.
  OUTPUT=$(nice -n 20 numactl --interleave=all sysbench memory \
    --memory-oper=read \
    --memory-block-size=1K \
    --memory-total-size=256G \
    --threads=48 run)

  # Extract the numeric MiB/sec (e.g. "210110.22") from the line:
  # "262143.98 MiB transferred (210110.22 MiB/sec)"
  RATE=$(echo "${OUTPUT}" | awk '/MiB transferred \(/ { 
    # Example line format:
    #   "262143.98 MiB transferred (210110.22 MiB/sec)"
    # We split on "(" and ")" and pull the numeric portion for MiB/sec.
    split($0, arr, "[()]")
    # arr[2] has something like "210110.22 MiB/sec"
    # Remove trailing " MiB/sec" from it.
    gsub(" MiB/sec", "", arr[2])
    print arr[2]
  }')

  echo "Run ${i}: ${RATE} MiB/sec"

  # Accumulate for average
  TOTAL=$(echo "${TOTAL} + ${RATE}" | bc -l)
done

# Compute average
AVERAGE=$(echo "${TOTAL} / ${NUM_RUNS}" | bc -l)

echo "Average throughput over ${NUM_RUNS} runs: ${AVERAGE} MiB/sec"

