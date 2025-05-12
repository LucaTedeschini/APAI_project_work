# High-Performance Neural Network Implementations

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

## Authors

Luca Tedeschini - University of Bologna