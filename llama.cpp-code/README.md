# Qwen2.5-Coder-32B-Instruct Perf Testing
We use [vLLM's benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) to test.

Mixed languages would be better, but since we're lazy, we use a ShareGPT coding dataset: https://huggingface.co/datasets/ajibawa-2023/Python-Code-23k-ShareGPT/resolve/main/Python-Code-23k-ShareGPT.json 
- 50 Prompts (~20min to run at 20 tok/s)
- seed=1 to keep things equal
- For AMD, we use the https://github.com/hjc4869/llama.cpp fork for best performance

Testing done 2024-12-31 w/ llama.cpp-hjc4869 b4398 on W7900 and llama.cpp build 4400 on RTX 3090 w/ standard compile flags.

```
# Repo
# git clone https://github.com/vllm-project/vllm

# Install dependencies
pip install datasets "numpy<2" pandas pillow transformers
# might need to just pip install vllm

# Grab Data
wget 'https://huggingface.co/datasets/ajibawa-2023/Python-Code-23k-ShareGPT/resolve/main/Python-Code-23k-ShareGPT.json'
```

## W7900
Non-Nvidia cards still perform worse w/ llama.cpp's FA implementation enabled so we don't use it.

Overall, using speculative decoding gives us ~60% higher throughput, and 40% lower TPOT, at the cost of 7.5% additional memory usage.


| Metric                          |   W7900 Q4_K_M |   W7900 Q4_K_M + 1.5B Q8 |   % Difference |
|:--------------------------------|---------------:|-------------------------:|---------------:|
| Memory Usage (GiB)              |          20.57 |                    22.12 |            7.5 |
| Successful requests             |          50    |                    50    |            0.0 |
| Benchmark duration (s)          |        1085.39 |                   678.21 |          -37.5 |
| Total input tokens              |        5926    |                  5926    |            0.0 |
| Total generated tokens          |       23110    |                 23204    |            0.4 |
| Request throughput (req/s)      |           0.05 |                     0.07 |           40.0 |
| Output token throughput (tok/s) |          21.29 |                    34.21 |           60.7 |
| Total Token throughput (tok/s)  |          26.75 |                    42.95 |           60.6 |
| Mean TTFT (ms)                  |         343.50 |                   344.16 |            0.2 |
| Median TTFT (ms)                |         345.69 |                   346.8  |            0.3 |
| P99 TTFT (ms)                   |         683.43 |                   683.85 |            0.1 |
| Mean TPOT (ms)                  |          46.09 |                    28.83 |          -37.4 |
| Median TPOT (ms)                |          45.97 |                    28.70 |          -37.6 |
| P99 TPOT (ms)                   |          47.70 |                    42.65 |          -10.6 |
| Mean ITL (ms)                   |          46.22 |                    28.48 |          -38.4 |
| Median ITL (ms)                 |          46.00 |                     0.04 |          -99.9 |
| P99 ITL (ms)                    |          48.79 |                   310.77 |          537.0 |


### W7900 Q4_K_M
```
# Server
~/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -ngl 99

# Benchmark
python benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "Qwen/Qwen2.5-Coder-32B-Instruct" --dataset-name sharegpt --dataset-path Python-Code-23k-ShareGPT.json --num-prompts 50 --max-concurrency 1 --seed 1

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  1085.39   
Total input tokens:                      5926      
Total generated tokens:                  23110     
Request throughput (req/s):              0.05      
Output token throughput (tok/s):         21.29     
Total Token throughput (tok/s):          26.75     
---------------Time to First Token----------------
Mean TTFT (ms):                          343.50    
Median TTFT (ms):                        345.69    
P99 TTFT (ms):                           683.43    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          46.09     
Median TPOT (ms):                        45.97     
P99 TPOT (ms):                           47.70     
---------------Inter-token Latency----------------
Mean ITL (ms):                           46.22     
Median ITL (ms):                         46.00     
P99 ITL (ms):                            48.79     
==================================================
```
- 20.567 GiB

### W7900 Q4_K_M + 1.5B Q8 Draft
```
# Server
~/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -md /models/gguf/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf --draft-max 24 --draft-min 1 --draft-p-min 0.6 -ngl 99 -ngld 99

# Benchmark
python benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "Qwen/Qwen2.5-Coder-32B-Instruct" --dataset-name sharegpt --dataset-path Python-Code-23k-ShareGPT.json --num-prompts 50 --max-concurrency 1 --seed 1

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  678.21    
Total input tokens:                      5926      
Total generated tokens:                  23204     
Request throughput (req/s):              0.07      
Output token throughput (tok/s):         34.21     
Total Token throughput (tok/s):          42.95     
---------------Time to First Token----------------
Mean TTFT (ms):                          344.16    
Median TTFT (ms):                        346.80    
P99 TTFT (ms):                           683.85    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.83     
Median TPOT (ms):                        28.70     
P99 TPOT (ms):                           42.65     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.48     
Median ITL (ms):                         0.04      
P99 ITL (ms):                            310.77    
==================================================
```
- 22.123 GiB

## RTX 3090

We test with and w/o FA and per usual, get some slight improvements w/ FA.

Overall, using speculative decoding gives us ~55% higher throughput, and 35% lower TPOT, at the cost of 9.5% additional memory usage.

| Metric                          |   RTX 3090 Q4_K_M |   RTX 3090 Q4_K_M + 1.5B Q8 |   % Difference |
|:--------------------------------|------------------:|----------------------------:|---------------:|
| Memory Usage (GiB)              |             20.20 |                       22.03 |            9.5 |
| Successful requests             |                50 |                          50 |            0.0 |
| Benchmark duration (s)          |            659.45 |                       419.7 |          -36.4 |
| Total input tokens              |              5926 |                        5926 |            0.0 |
| Total generated tokens          |             23447 |                       23123 |           -1.4 |
| Request throughput (req/s)      |              0.08 |                        0.12 |           50.0 |
| Output token throughput (tok/s) |             35.56 |                       55.09 |           54.9 |
| Total Token throughput (tok/s)  |             44.54 |                       69.21 |           55.4 |
| Mean TTFT (ms)                  |            140.01 |                      141.43 |            1.0 |
| Median TTFT (ms)                |             97.17 |                       97.92 |            0.8 |
| P99 TTFT (ms)                   |            373.87 |                      407.96 |            9.1 |
| Mean TPOT (ms)                  |             27.85 |                       18.23 |          -34.5 |
| Median TPOT (ms)                |             27.80 |                       17.96 |          -35.4 |
| P99 TPOT (ms)                   |             28.73 |                       28.14 |           -2.1 |
| Mean ITL (ms)                   |             27.82 |                       17.83 |          -35.9 |
| Median ITL (ms)                 |             27.77 |                        0.02 |          -99.9 |
| P99 ITL (ms)                    |             29.34 |                      160.18 |          445.9 |


### RTX 3090 Q4_K_M
```
# Server
~/llama.cpp/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -ngl 99

# Benchmark
python benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "Qwen/Qwen2.5-Coder-32B-Instruct" --dataset-name sharegpt --dataset-path Python-Code-23k-ShareGPT.json --num-prompts 50 --max-concurrency 1 --seed 1

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  669.26    
Total input tokens:                      5926      
Total generated tokens:                  23120     
Request throughput (req/s):              0.07      
Output token throughput (tok/s):         34.55     
Total Token throughput (tok/s):          43.40     
---------------Time to First Token----------------
Mean TTFT (ms):                          140.44    
Median TTFT (ms):                        95.37     
P99 TTFT (ms):                           383.33    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.62     
Median TPOT (ms):                        28.63     
P99 TPOT (ms):                           29.33     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.64     
Median ITL (ms):                         28.63     
P99 ITL (ms):                            31.17     
==================================================
```
- 20.213 GiB

### RTX 3090 Q4_K_M w/ FA
```
# Server
~/llama.cpp/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -ngl 99 -fa

# Benchmark
python benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "Qwen/Qwen2.5-Coder-32B-Instruct" --dataset-name sharegpt --dataset-path Python-Code-23k-ShareGPT.json --num-prompts 50 --max-concurrency 1 --seed 1

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  659.45    
Total input tokens:                      5926      
Total generated tokens:                  23447     
Request throughput (req/s):              0.08      
Output token throughput (tok/s):         35.56     
Total Token throughput (tok/s):          44.54     
---------------Time to First Token----------------
Mean TTFT (ms):                          140.01    
Median TTFT (ms):                        97.17     
P99 TTFT (ms):                           373.87    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.85     
Median TPOT (ms):                        27.80     
P99 TPOT (ms):                           28.73     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.82     
Median ITL (ms):                         27.77     
P99 ITL (ms):                            29.34     
==================================================
```
- 20.119 GiB

### RTX 3090 Q4_K_M + 1.5B Q8 Draft
```
# Server
~/llama.cpp-hjc4869/build/bin/llama-server -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -md /models/gguf/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf --draft-max 24 --draft-min 1 --draft-p-min 0.6 -ngl 99 -ngld 99

# Benchmark
python benchmarks/benchmark_serving.py --backend openai-chat --host localhost --port 8080 --endpoint='/v1/chat/completions' --model "Qwen/Qwen2.5-Coder-32B-Instruct" --dataset-name sharegpt --dataset-path Python-Code-23k-ShareGPT.json --num-prompts 50 --max-concurrency 1 --seed 1

============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  419.70    
Total input tokens:                      5926      
Total generated tokens:                  23123     
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         55.09     
Total Token throughput (tok/s):          69.21     
---------------Time to First Token----------------
Mean TTFT (ms):                          141.43    
Median TTFT (ms):                        97.92     
P99 TTFT (ms):                           407.96    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.23     
Median TPOT (ms):                        17.96     
P99 TPOT (ms):                           28.14     
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.83     
Median ITL (ms):                         0.02      
P99 ITL (ms):                            160.18    
==================================================
```
- 22.031 GiB

## W7900 vs RTX 3090

When using the best results, the RTX 3090 is significantly faster than the W7900:
- \>60% higher throughput
- \>70% lower median TTFT
- ~37% lower TPOT

| Metric                          |   W7900 Q4_K_M + 1.5B Q8 |   RTX 3090 Q4_K_M + 1.5B Q8 |   % Difference |
|:--------------------------------|-------------------------:|----------------------------:|---------------:|
| Memory Usage (GiB)              |                    22.12 |                       22.03 |           -0.4 |
| Successful requests             |                       50 |                          50 |            0.0 |
| Benchmark duration (s)          |                   678.21 |                      419.70 |          -38.1 |
| Total input tokens              |                     5926 |                        5926 |            0.0 |
| Total generated tokens          |                    23204 |                       23123 |           -0.3 |
| Request throughput (req/s)      |                     0.07 |                        0.12 |           71.4 |
| Output token throughput (tok/s) |                    34.21 |                       55.09 |           61.0 |
| Total Token throughput (tok/s)  |                    42.95 |                       69.21 |           61.1 |
| Mean TTFT (ms)                  |                   344.16 |                      141.43 |          -58.9 |
| Median TTFT (ms)                |                   346.8  |                       97.92 |          -71.8 |
| P99 TTFT (ms)                   |                   683.85 |                      407.96 |          -40.3 |
| Mean TPOT (ms)                  |                    28.83 |                       18.23 |          -36.8 |
| Median TPOT (ms)                |                    28.7  |                       17.96 |          -37.4 |
| P99 TPOT (ms)                   |                    42.65 |                       28.14 |          -34.0 |
| Mean ITL (ms)                   |                    28.48 |                       17.83 |          -37.4 |
| Median ITL (ms)                 |                     0.04 |                        0.02 |          -50.0 |
| P99 ITL (ms)                    |                   310.77 |                      160.18 |          -48.5 |

## llama-bench
Just for reference. Note that pp512 is 2X faster on the 3090. This makes sense as the CUDA backend primarily uses INT8 and the 3090 has 284 INT8 TOPS vs the W7900's 122.6 FP16 TFLOPS.

The W7900 864.0 GB/s MBW vs the 3090 with 936.2 GB/s (almost the same), so the ROCm backend is significantly less efficient than the CUDA backend.

See also: https://www.reddit.com/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/

### W7900
```
❯ ~/llama.cpp-hjc4869/build/bin/llama-bench -m /models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro W7900, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | ROCm       |  99 |         pp512 |        663.44 ± 1.50 |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | ROCm       |  99 |         tg128 |         23.00 ± 0.06 |

build: a0c09b1c (4398)
```

### RTX 3090
```
❯ CUDA_VISIBLE_DEVICES=1 build/bin/llama-bench -m /models/llm/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |         pp512 |       1294.03 ± 7.76 |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |         tg128 |         37.71 ± 0.15 |

build: 6e1531ac (4400)

❯ CUDA_VISIBLE_DEVICES=1 build/bin/llama-bench -m /models/llm/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         pp512 |      1335.70 ± 25.01 |
| qwen2 32B Q4_K - Medium        |  18.48 GiB |    32.76 B | CUDA       |  99 |  1 |         tg128 |         38.34 ± 0.15 |

build: 6e1531ac (4400)
```


# Future
Do broader testing with? (maybe be overfitted)
- https://github.com/amazon-science/mxeval
- https://llm-tracker.info/evals/EvalPlus

