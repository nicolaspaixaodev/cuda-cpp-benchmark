# CUDA vs. CPU Latency & Throughput Benchmark

C++/CUDA benchmark comparing sequential CPU execution against parallel GPU execution on NVIDIA Turing architecture (`sm_75`), highlighting GPU warm-up, architecture-specific compilation, and compute-bound vs. memory-bound workloads.

## Hardware & Environment
- CPU: Intel Core i7-11800H
- GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM, sm_75)
- OS / Compiler: Ubuntu 24.04 LTS (WSL2), CUDA Toolkit 12.6

## Benchmark Results
- Vector Addition (10M elements, cold start): CPU 23.732 ms | GPU 178.634 ms (0.1x)
- Heavy Transcendental Math (2M elements, warm-up + -arch=sm_75): CPU 11730.512 ms | GPU 31.510 ms (372.3x speedup)

## Build & Run
Run `make` and then `./benchmark_pesado`.
