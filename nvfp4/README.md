# NVFP4 and alternatives on Blackwell
Testing done on a single 600W RTX PRO 6000 Workstation (temps <70C):
- AMD EPYC 9275F
- Linux 6.17.0-4-cachyos
- nvidia-open-dkms 580.95.05-2
- cuda_13.0.r13.0/compiler.36424714_0 

## Testing
I'm testing with [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) since the original Qwen3 30B-A3B NVFP4 quant didn't load, but it's a good standard model to test with and there are lots of quants and even our bonus (EAGLE3 head...)

### llama.cpp
Llama.cpp is the best solution for bs=1/c=1 , but even with multi-user optimizations gets quickly overwhelmed.

- Q4_K_XL: [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

### Run
```
# Build first
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j48

# Run - 95% memory
CUDA_VISIBLE_DEVICES=0 build/bin/llama-server -fa 1 -m /models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8000 -np 32 -c 700000

```
- https://github.com/ggml-org/llama.cpp/discussions/9041

### SGLang
SGLang is quite good, although in general it's Blackwell support is somewhat iffy. On older architectures, it's W8A8-INT8 performance is somewhat bonkers, but INT8 doesn't even work on Blackwell OOTB... 

- FP8 (auto): Load full model and `--quantization fp8`
- FP8 Dynamic: [RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic](RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic) - we have to do our own patch
  - https://github.com/sgl-project/sglang/issues/7482
  - https://github.com/sgl-project/sglang/pull/9403
  - https://github.com/sgl-project/sglang/issues/4524
- W8A8-INT8: [RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8)
  - Implemented a fix but so slow to not be worth finishing benchmarks
- W4A16: [RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16)

```
python -m sglang.launch_server --port 8000 --model meta-llama/Llama-3.1-8B-Instruct --quantization fp8
```

FP8 Fixes
```
# sglang/srt/layers/quantization/fp8_utils.py
84a85,89
>     # CUTLASS FP8 kernels are currently unstable on Blackwell (SM120+) and
>     # trigger cutlass::Status failures during CUDA graph capture. Force a
>     # fallback path until the upstream fix lands.
>     if major >= 12:
>         return False
```

INT8 Fixes
```
# sgl_kernel/gemm.py
15,22c15,85
<     return torch.ops.sgl_kernel.int8_scaled_mm.default(
<         mat_a,
<         mat_b,
<         scales_a,
<         scales_b,
<         out_dtype,
<         bias,
<     )
---
>     try:
>         return torch.ops.sgl_kernel.int8_scaled_mm.default(
>             mat_a,
>             mat_b,
>             scales_a,
>             scales_b,
>             out_dtype,
>             bias,
>         )
>     except (RuntimeError, NotImplementedError) as exc:
>         message = str(exc)
>         if "No implemented int8_scaled_mm" not in message:
>             raise
>     # Try PyTorch's fused scaled GEMM if available (uses cuBLASLt under the
>     # hood and is much faster than dequantizing).
>     if hasattr(torch, "_scaled_mm"):
>         a_scales = scales_a.to(torch.float32)
>         if a_scales.ndim > 1:
>             # Accept (M, 1) or similar by squeezing the redundant dim.
>             if a_scales.shape[-1] == 1:
>                 a_scales = a_scales.squeeze(-1)
>             else:
>                 a_scales = a_scales.reshape(a_scales.shape[0], -1)
>                 if a_scales.shape[-1] == 1:
>                     a_scales = a_scales.squeeze(-1)
>                 else:
>                     a_scales = a_scales.reshape(-1)
>         else:
>             a_scales = a_scales.reshape(-1)
>
>         b_scales = scales_b.to(torch.float32)
>         if b_scales.ndim > 1:
>             if b_scales.shape[-1] == 1:
>                 b_scales = b_scales.squeeze(-1)
>             elif b_scales.shape[0] == 1:
>                 b_scales = b_scales.squeeze(0)
>             else:
>                 b_scales = b_scales.reshape(-1)
>         else:
>             b_scales = b_scales.reshape(-1)
>
>         try:
>             return torch._scaled_mm(
>                 mat_a,
>                 mat_b,
>                 out_dtype=out_dtype,
>                 scale_a=a_scales.contiguous(),
>                 scale_b=b_scales.contiguous(),
>                 bias=bias,
>             )
>         except (RuntimeError, NotImplementedError):
>             pass
>     # Fall back to dequantize-and-matmul when the custom CUTLASS kernel
>     # is unavailable on this architecture (e.g., Blackwell SM120).
>     mat_a_f32 = mat_a.to(torch.float32)
>     scales_a_f32 = scales_a.to(torch.float32)
>     while scales_a_f32.ndim < mat_a_f32.ndim:
>         scales_a_f32 = scales_a_f32.unsqueeze(-1)
>     mat_a_f32 = mat_a_f32 * scales_a_f32
>
>     mat_b_f32 = mat_b.to(torch.float32)
>     scales_b_f32 = scales_b.to(torch.float32).reshape(-1)
>     if scales_b_f32.numel() == mat_b_f32.shape[1]:
>         mat_b_f32 = mat_b_f32 * scales_b_f32.view(1, -1)
>     else:
>         mat_b_f32 = mat_b_f32 * scales_b_f32.view(1, 1)
>
>     output = mat_a_f32.matmul(mat_b_f32)
>     if bias is not None:
>         output = output + bias.to(output.dtype)
>     return output.to(out_dtype)
```

## TensorRT:
People are always like Blackwell is all about it's FP4 performance but uh, AFAICT, even using the Nvidia models w/ TensorRT (the only NVFP4 optimized platform), er the performance is nothing to write home about 

- NVFP8: [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- NVFP4: [nvidia/Llama-3.1-8B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP4)

### Usage 
Using `nvidia-container-toolkit`:
```
# Setup
sudo docker run --rm -it --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 \
  -v /data/huggingface/hub:/root/.cache/huggingface/hub \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0

# Serving
trtserve-llm nvidia/Llama-3.1-8B-Instruct-FP8 --host 0.0.0.0
trtserve-llm nvidia/Llama-3.1-8B-Instruct-FP4 --host 0.0.0.0
```

## vLLM
vLLM is pretty neck and neck w/ SGLang on throughput and actually has slightly better OOTB latency atm. These trade back and forth - either is fine, which you choose will probably come down to model and platform-specific  support.

- FP8 Dynamic: [RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic](RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic)
- W8A8-INT8: broken for Blackwell! https://github.com/vllm-project/vllm/issues/21097
- W4A16: [RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16)

```
vllm serve RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
vllm serve RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16

```


## Performance
Output from `analyze-bench.py` - run it w/ `rich` for colors
```
Throughput metrics: higher is better | Latency metrics (TTFT/TPOT): lower is better

╭────────────────────┬───────┬─────────┬────────┬─────────┬─────────┬───────┬───────┬───────┬───────┬───────┬───────╮
│                    │       │ Prefill │ Decode │   Total │ Max Out │  TTFT │  TTFT │  TTFT │  TPOT │  TPOT │  TPOT │
│ Config             │ Req/s │   Tok/s │  Tok/s │   Tok/s │   Tok/s │  mean │   med │   p99 │  mean │   med │   p99 │
├────────────────────┼───────┼─────────┼────────┼─────────┼─────────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ llama.cpp.q4_k_m   │  1.65 │ 1683.45 │ 207.16 │ 1890.61 │  223.00 │ 74.17 │ 75.75 │ 85.71 │  4.36 │  4.22 │  8.40 │
│ sglang.fp8-auto    │  1.15 │ 1173.85 │ 142.83 │ 1316.68 │  146.00 │ 54.88 │ 55.31 │ 55.79 │  6.61 │  6.62 │  6.62 │
│ sglang.fp8-dynamic │  1.04 │ 1065.99 │ 130.29 │ 1196.28 │  132.00 │ 55.91 │ 56.30 │ 57.13 │  7.28 │  7.29 │  7.29 │
│ sglang.w4a16       │  1.56 │ 1590.93 │ 194.85 │ 1785.78 │  204.00 │ 53.69 │ 54.10 │ 54.79 │  4.74 │  4.75 │  4.76 │
│ trt.fp8            │  0.59 │  605.67 │  74.33 │  680.01 │   76.00 │ 39.94 │ 40.24 │ 40.76 │ 13.24 │ 13.24 │ 13.27 │
│ trt.nvfp4          │  0.60 │  608.22 │  74.38 │  682.61 │   76.00 │ 30.91 │ 31.05 │ 31.31 │ 13.30 │ 13.30 │ 13.34 │
│ vllm.fp8-dynamic   │  0.77 │  789.55 │  94.90 │  884.45 │   98.00 │ 34.94 │ 35.12 │ 36.43 │ 10.34 │ 10.34 │ 10.36 │
│ vllm.w4a16         │  1.52 │ 1549.83 │ 189.81 │ 1739.64 │  196.00 │ 49.09 │ 49.39 │ 50.30 │  4.92 │  4.92 │  4.96 │
╰────────────────────┴───────┴─────────┴────────┴─────────┴─────────┴───────┴───────┴───────┴───────┴───────┴───────╯

                                                     Concurrency: 4
╭────────────────────┬───────┬─────────┬────────┬─────────┬─────────┬────────┬────────┬────────┬───────┬───────┬───────╮
│                    │       │ Prefill │ Decode │   Total │ Max Out │   TTFT │   TTFT │   TTFT │  TPOT │  TPOT │  TPOT │
│ Config             │ Req/s │   Tok/s │  Tok/s │   Tok/s │   Tok/s │   mean │    med │    p99 │  mean │   med │   p99 │
├────────────────────┼───────┼─────────┼────────┼─────────┼─────────┼────────┼────────┼────────┼───────┼───────┼───────┤
│ llama.cpp.q4_k_m   │  2.98 │ 3045.20 │ 374.73 │ 3419.93 │  514.00 │ 241.75 │ 215.02 │ 330.95 │  9.80 │  7.52 │ 20.86 │
│ sglang.fp8-auto    │  3.99 │ 4069.51 │ 501.77 │ 4571.29 │  516.00 │  23.94 │  23.90 │  45.97 │  7.71 │  7.72 │  7.84 │
│ sglang.fp8-dynamic │  4.04 │ 4119.66 │ 493.32 │ 4612.98 │  512.00 │  24.70 │  24.40 │  27.19 │  7.89 │  7.93 │  8.01 │
│ sglang.w4a16       │  5.86 │ 5977.68 │ 732.11 │ 6709.79 │  764.00 │  23.81 │  16.98 │ 222.63 │  5.22 │  5.22 │  5.30 │
│ trt.fp8            │  1.09 │ 1111.39 │ 138.80 │ 1250.19 │  144.00 │  51.90 │  54.39 │  62.39 │ 28.56 │ 28.58 │ 28.81 │
│ trt.nvfp4          │  1.11 │ 1135.33 │ 138.15 │ 1273.48 │  145.00 │  52.77 │  62.15 │  63.30 │ 28.49 │ 28.60 │ 28.91 │
│ vllm.fp8-dynamic   │  3.00 │ 3058.79 │ 367.81 │ 3426.60 │  380.00 │  21.26 │  21.20 │  28.21 │ 10.64 │ 10.64 │ 10.66 │
│ vllm.w4a16         │  5.89 │ 6008.29 │ 735.86 │ 6744.15 │  757.00 │  17.14 │  17.27 │  21.75 │  5.25 │  5.25 │  5.26 │
╰────────────────────┴───────┴─────────┴────────┴─────────┴─────────┴────────┴────────┴────────┴───────┴───────┴───────╯

                                                      Concurrency: 8
╭────────────────────┬───────┬──────────┬─────────┬──────────┬─────────┬────────┬────────┬────────┬───────┬───────┬───────╮
│                    │       │  Prefill │  Decode │    Total │ Max Out │   TTFT │   TTFT │   TTFT │  TPOT │  TPOT │  TPOT │
│ Config             │ Req/s │    Tok/s │   Tok/s │    Tok/s │   Tok/s │   mean │    med │    p99 │  mean │   med │   p99 │
├────────────────────┼───────┼──────────┼─────────┼──────────┼─────────┼────────┼────────┼────────┼───────┼───────┼───────┤
│ llama.cpp.q4_k_m   │  2.65 │  2705.48 │  333.27 │  3038.75 │  456.00 │ 310.03 │ 326.53 │ 585.15 │ 21.67 │ 20.36 │ 39.22 │
│ sglang.fp8-auto    │  7.46 │  7618.19 │  923.81 │  8542.00 │  981.00 │  27.91 │  29.62 │  34.02 │  8.25 │  8.27 │  8.92 │
│ sglang.fp8-dynamic │  7.35 │  7496.84 │  908.12 │  8404.96 │  968.00 │  28.32 │  25.35 │  34.77 │  8.37 │  8.42 │  8.55 │
│ sglang.w4a16       │ 10.88 │ 11106.95 │ 1360.31 │ 12467.26 │ 1417.00 │  20.89 │  21.78 │  26.29 │  5.63 │  5.66 │  5.72 │
│ trt.fp8            │  7.62 │  7780.11 │  943.09 │  8723.21 │ 1144.00 │  22.73 │  22.18 │  27.45 │  7.52 │  6.97 │ 23.89 │
│ trt.nvfp4          │  8.00 │  8169.26 │  994.33 │  9163.59 │ 1240.00 │  26.85 │  21.27 │  45.61 │  7.05 │  6.42 │ 25.35 │
│ vllm.fp8-dynamic   │  5.71 │  5829.43 │  708.46 │  6537.89 │  736.00 │  22.29 │  22.17 │  31.72 │ 10.86 │ 10.86 │ 10.90 │
│ vllm.w4a16         │ 10.85 │ 11070.41 │ 1355.84 │ 12426.24 │ 1401.00 │  20.24 │  20.78 │  26.69 │  5.64 │  5.67 │  5.69 │
╰────────────────────┴───────┴──────────┴─────────┴──────────┴─────────┴────────┴────────┴────────┴───────┴───────┴───────╯

                                                      Concurrency: 16
╭────────────────────┬───────┬──────────┬─────────┬──────────┬─────────┬────────┬────────┬─────────┬───────┬───────┬───────╮
│                    │       │  Prefill │  Decode │    Total │ Max Out │   TTFT │   TTFT │    TTFT │  TPOT │  TPOT │  TPOT │
│ Config             │ Req/s │    Tok/s │   Tok/s │    Tok/s │   Tok/s │   mean │    med │     p99 │  mean │   med │   p99 │
├────────────────────┼───────┼──────────┼─────────┼──────────┼─────────┼────────┼────────┼─────────┼───────┼───────┼───────┤
│ llama.cpp.q4_k_m   │  2.56 │  2609.62 │  320.94 │  2930.56 │  448.00 │ 438.32 │ 374.44 │ 1259.08 │ 48.85 │ 46.90 │ 94.06 │
│ sglang.fp8-auto    │ 13.76 │ 14040.79 │ 1720.70 │ 15761.49 │ 1824.00 │  37.20 │  39.50 │   42.83 │  8.87 │  8.89 │  9.11 │
│ sglang.fp8-dynamic │ 13.42 │ 13699.42 │ 1686.21 │ 15385.63 │ 1736.00 │  35.70 │  35.24 │   45.16 │  9.10 │  9.12 │  9.38 │
│ sglang.w4a16       │ 18.60 │ 18983.13 │ 2324.94 │ 21308.06 │ 2402.00 │  29.13 │  31.65 │   37.70 │  6.57 │  6.57 │  7.75 │
│ trt.fp8            │ 17.04 │ 17392.52 │ 2150.50 │ 19543.02 │ 2192.00 │  27.81 │  27.00 │   42.47 │  7.17 │  7.16 │  7.23 │
│ trt.nvfp4          │ 18.35 │ 18724.64 │ 2290.70 │ 21015.34 │ 2368.00 │  29.79 │  27.53 │   63.60 │  6.64 │  6.63 │  6.72 │
│ vllm.fp8-dynamic   │ 10.74 │ 10959.66 │ 1338.08 │ 12297.74 │ 1390.00 │  34.05 │  34.01 │   53.60 │ 11.44 │ 11.48 │ 11.49 │
│ vllm.w4a16         │ 18.56 │ 18946.49 │ 2320.45 │ 21266.93 │ 2390.00 │  31.36 │  32.97 │   51.27 │  6.52 │  6.56 │  6.63 │
╰────────────────────┴───────┴──────────┴─────────┴──────────┴─────────┴────────┴────────┴─────────┴───────┴───────┴───────╯

                                                      Concurrency: 32
╭────────────────────┬───────┬──────────┬─────────┬──────────┬─────────┬────────┬────────┬─────────┬───────┬───────┬───────╮
│                    │       │  Prefill │  Decode │    Total │ Max Out │   TTFT │   TTFT │    TTFT │  TPOT │  TPOT │  TPOT │
│ Config             │ Req/s │    Tok/s │   Tok/s │    Tok/s │   Tok/s │   mean │    med │     p99 │  mean │   med │   p99 │
├────────────────────┼───────┼──────────┼─────────┼──────────┼─────────┼────────┼────────┼─────────┼───────┼───────┼───────┤
│ llama.cpp.q4_k_m   │  6.82 │  6962.35 │  856.84 │  7819.20 │ 1952.00 │ 739.09 │ 428.11 │ 2554.77 │ 34.12 │ 32.92 │ 65.97 │
│ sglang.fp8-auto    │ 21.96 │ 22409.88 │ 2705.51 │ 25115.40 │ 2932.00 │ 103.82 │  42.84 │  309.14 │ 10.35 │ 10.85 │ 11.19 │
│ sglang.fp8-dynamic │ 21.70 │ 22141.78 │ 2698.06 │ 24839.84 │ 2847.00 │ 107.47 │  57.27 │  316.26 │ 10.70 │ 10.80 │ 11.65 │
│ sglang.w4a16       │ 26.94 │ 27498.19 │ 3367.81 │ 30866.00 │ 3390.00 │  95.33 │  48.22 │  270.54 │  8.57 │  8.98 │ 10.67 │
│ trt.fp8            │ 30.59 │ 31214.81 │ 3865.77 │ 35080.58 │ 3936.00 │  37.04 │  28.38 │   86.05 │  7.89 │  7.94 │  7.99 │
│ trt.nvfp4          │ 32.77 │ 33444.34 │ 4064.82 │ 37509.16 │ 4209.00 │  39.75 │  31.96 │   84.06 │  7.37 │  7.36 │  9.20 │
│ vllm.fp8-dynamic   │ 19.72 │ 20129.64 │ 2466.74 │ 22596.38 │ 2560.00 │  45.13 │  47.92 │   55.27 │ 12.38 │ 12.45 │ 12.56 │
│ vllm.w4a16         │ 26.43 │ 26970.73 │ 3303.21 │ 30273.94 │ 3332.00 │  44.73 │  46.55 │   61.96 │  9.15 │  9.23 │  9.40 │
╰────────────────────┴───────┴──────────┴─────────┴──────────┴─────────┴────────┴────────┴─────────┴───────┴───────┴───────╯

```
