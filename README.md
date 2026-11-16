# Jetson CUDA Sobel Edge Detector (`sobel_ops`)

A portable **CUDA C++ & Python (`pybind11`)** extension implementing a 2D Sobel edge detection filter using **shared-memory tiling with halo cells**. Designed for **NVIDIA Jetson Orin Nano (`sm_87`)**, with automatic CMake fallback to a pure C++ backend on machines without an NVIDIA GPU.

## Features
* **Shared-Memory Tiling (`16x16` + 1px Halo):** Loads `18x18` blocks cooperatively into `__shared__` memory to minimize redundant global memory reads.
* **Zero-Boilerplate Python API:** Exposes `sobel_filter()` directly to NumPy `float32` arrays via `pybind11`.
* **Cross-Platform Build System:** CMake automatically detects `nvcc` and compiles the CUDA backend when available, or falls back cleanly to `src/sobel_cpu.cpp` on non-GPU systems.

## Benchmark (NVIDIA Jetson Orin Nano)
Tested on a `2048 x 2048` single-precision (`float32`) image:

| Implementation | Execution Time | Speedup | Max Absolute Error |
| :--- | :--- | :--- | :--- |
| **CUDA Shared-Memory Kernel (`sm_87`)** | **17.50 ms** | **8.60x** | `2.38e-07` |
| NumPy Vectorized Reference | 150.54 ms | 1.00x | Baseline |

## Build & Run

### Prerequisites
* CMake >= 3.20
* C++17 compiler (`g++` / `clang++`)
* Python 3 with `pybind11` and `numpy` (`pip install pybind11 numpy`)
* *(Optional)* NVIDIA CUDA Toolkit (`nvcc`) for GPU acceleration

### Compile & Verify
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j\$(nproc)
python3 python/benchmark.py
```
