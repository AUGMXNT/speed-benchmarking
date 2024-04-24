# 2024-02 Testing

For results, see: https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=1652827441

## Systems

### 7900 XT on Ryzen 5600G
  - Ubuntu 22.04.4 LTS
  - Linux rocm 6.5.0-21-generic #21~22.04.1-Ubuntu
  - ROCm 6.0.0.60000 (latest package from AMD's repo https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/ubuntu.html)

```
$ dpkg -l rocm
+++-==============-====================-============-======================================================
ii  rocm           6.0.0.60000-91~22.04 amd64        Radeon Open Compute (ROCm) software stack meta package
```

### RTX 3090 and RTX 4090 on Ryzen 5950X
- Arch Linux
- Linux ai 6.7.6-arch1-1
- nvidia-dkms 545.29.06-4
- CUDA 12.1 (mamba)
