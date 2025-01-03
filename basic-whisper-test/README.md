

# CPU

## AMD EPYC 9274F
Can't make it run on all CPUs?

## Intel Core Ultra 7 258V
Audio duration: 62.45 seconds
Audio size: (999280,)
Sampling rate: 16000 Hz
Inference time: 22.90 seconds
Real-time factor: 0.37x
Real-time X: 2.73x


# GPU

## RTX 3050
Audio duration: 62.45 seconds
Audio size: (999280,)
Sampling rate: 16000 Hz
Inference time: 2.06 seconds
Real-time factor: 0.03x
Real-time X: 30.34x

## W7900
Audio duration: 62.45 seconds
Audio size: (999280,)
Sampling rate: 16000 Hz
Inference time: 1.88 seconds
Real-time factor: 0.03x
Real-time X: 33.21x

## Arc
/home/lhl/miniforge3/lib/python3.12/site-packages/transformers/pytorch_utils.py:341: UserWarning: Aten Op fallback from XPU to CPU happends. This may have performance implications. If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1`  (Triggered internally at /pytorch/build/aten/src/ATen/xpu/RegisterXPU.cpp:7614.)
  return torch.isin(elements, test_elements)

Audio duration: 62.45 seconds
Audio size: (999280,)
Sampling rate: 16000 Hz
Inference time: 9.21 seconds
Real-time factor: 0.15x
Real-time X: 6.78x
