# High-Performance Neural Network Implementations

## Project Overview

This project implements a sparsely connected Feed-Forward Neural Network (FFNN) using two parallel computing approaches:
1. OpenMP for multi-core CPU parallelization
2. CUDA for GPU acceleration

This project is part of the Architecture and Platform for Artificial Intelligence exam.

### Network Architecture

The neural network has the following key characteristics:
- Input layer size: N neurons
- Total layers: K
- Connectivity reach: R (each node connects to R previous nodes)
- Uses sigmoid activation function
- Includes a bias term

### Parallelization Strategies

#### OpenMP Implementation
- Parallelizes computation across nodes within each layer
- Uses `#pragma omp parallel for` to distribute workload
- Supports multi-threading on shared-memory CPU architectures

#### CUDA Implementation
- Implements two kernel versions:
  1. Shared memory kernel
  2. Global memory kernel
- Uses double-buffering strategy for layer computations
- Optimizes memory access patterns for GPU computation

## Compilation and Running

### OpenMP Version

#### Compilation
```bash
cd openMP
gcc -std=c99 -Wall -Wpedantic -fopenmp -Iinclude src/main.c src/network.c src/utilities.c -o openMP -lm
```

or

```bash
cd openMP
mkdir build
cd build
cmake ..
make
```

#### Running
```bash
./openMP [N] [K] [machine_output]
```
- `N`: Number of neurons in the first layer
- `K`: Total number of layers
- `machine_output`: 1 for machine-readable output, 0 for human-readable (default: 1)

### CUDA Version

#### Compilation
```bash
cd CUDA
mkdir build
nvcc -Iinclude src/main.cu -o build/main
```

#### Running
```bash
cd build
./main [N] [K] [machine_output]
```
- Same parameters as OpenMP version

## Analysis
Refer to `Report.pdf` for a more accurate analysis of the performances of both implementations

## Requirements

- GCC with OpenMP support
- NVIDIA CUDA Toolkit
- CMake (optional, for alternative build method)

## Authors

Luca Tedeschini - University of Bologna
