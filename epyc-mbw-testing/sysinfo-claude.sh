#!/bin/bash

# Exit on error
set -e

echo "============================"
echo "System Information Collection"
echo "============================"
echo "Date: $(date)"
echo

echo "============================"
echo "Basic System Information (uname)"
echo "============================"
echo "Kernel: $(uname -a)"
echo "OS Release: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
echo

echo "============================"
echo "System Information (inxi)"
echo "============================"
inxi -Fxxxz
echo

echo "============================"
echo "Basic Hardware Information (inxi)"
echo "============================"
inxi -M
echo

echo "============================"
echo "CPU Information (inxi)"
echo "============================"
inxi -C
echo

echo "============================"
echo "Memory Information (inxi)"
echo "============================"
inxi -I
echo

echo "============================"
echo "CPU Information (lscpu)"
echo "============================"
lscpu
echo

echo "============================"
echo "CPU Cache Information (lscpu -C)"
echo "============================"
lscpu -C
echo

echo "============================"
echo "NUMA/Topology Information (lstopo)"
echo "============================"
lstopo | cat
echo

echo "============================"
echo "NUMA Hardware Information (numactl)"
echo "============================"
numactl --hardware
echo

echo "============================"
echo "Processor Information (dmidecode)"
echo "============================"
sudo dmidecode -t processor
echo

echo "============================"
echo "Cache Information (dmidecode)"
echo "============================"
sudo dmidecode -t cache
echo

echo "============================"
echo "Memory Information (dmidecode)"
echo "============================"
sudo dmidecode -t memory
echo

echo "============================"
echo "CPU Information (hwinfo)"
echo "============================"
sudo hwinfo --cpu
echo

echo "============================"
echo "Memory Information (hwinfo)"
echo "============================"
sudo hwinfo --memory
echo

echo "============================"
echo "BIOS Information (hwinfo)"
echo "============================"
sudo hwinfo --bios
echo

echo "============================"
echo "Memory Information (free)"
echo "============================"
free -h
echo

echo "============================"
echo "CPU Vulnerabilities"
echo "============================"
grep . /sys/devices/system/cpu/vulnerabilities/*
echo

echo "============================"
echo "Done collecting system information"
echo "============================"
