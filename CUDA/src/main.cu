/****************************************************************************
*   CUDA Multi Layer Neural Network Implementation.
*
*
*   On linux, compile using nvcc on cuda capable devices
*   Example:
*       $ mkdir build
*       $ nvcc -Iinclude src/main.cu -o build/main
*
*    Or, if you have `cmake` and `make` avaiable
*        - path/to/root/folder$ mkdir build; cd build
*        - path/to/root/folder/build$ cmake ..
*        - path/to/root/folder/build$ make
*    This will create the executable.
*
*
*   Run the executable with:
*   ./cuda [N] [K] [machine output]
*
*   (N = first layer n° neurons)
*   (K = number of layers)
*   (machine output = 1 output machine readable, 0 output human readable. Default: 1)
****************************************************************************/


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string.h>

#include "hpc.h"


/***************************
 * 
 *  CONSTANT DEFINITION
 * 
****************************/

// Define the "spread index" between layers. Each node is linked to the previous R nodes.
#define R            3

// Define the block dimension for the CUDA kernels
#define BLK_DIM      1024

// Define the BIAS term
#define BIAS         0.2f


__device__ __forceinline__ float sigmoid(float x) {
    /* Sigmoid activation function */
    return 1.0f / (1.0f + expf(-x));
}


inline int layer_size(int N0, int layer) {
    /* Returns the layer-th layer size, given N */
    return N0 - layer * (R - 1);
}


__global__ void forwardpass_shared(
    const float* __restrict__ current_layer,
    float* __restrict__ next_layer,
    const float* __restrict__ W,
    const int N,
    const int out_size
) {
    /* Computes the forward pass using shared memory.
    
    It works by loading the necessary input element into the shared memory before doing
    the forward pass. To do so, it allocate a shared memory region (`s_input`) of dimension
    BLK_DIM + 2 * (R - 1). It must contain at least N elements due to the double
    buffer strategy implemented.
    A more detailed version can be found in the report.pdf in the root folder of the
    project.

    @param current_layer Pointer to the input layer data (activations from the previous layer) in global memory.
    @param next_layer Pointer to the output layer data (activations for the current layer being computed) in global memory.
    @param W Pointer to the weights array in global memory.
    @param N Size (number of elements) of the first input layer (`current_layer`). Used for bounds checking during loading.
    @param out_size Size (number of elements) of the output layer (`next_layer`).
    */

    // shared memory region declaration
    __shared__ float s_input[BLK_DIM + 2 * (R - 1)];

    // halo is being used to load the correct number of elements into the shared memory
    int halo = R - 1;


    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;
    int base_index = blockIdx.x * blockDim.x;

    // load the data into the shared memory
    if (global_index < out_size) {
        s_input[local_index] = current_layer[global_index];
    }
    if (local_index < halo) {
        // since out_size < N, but we need additional `halo` elements to corretly do the
        // computation, some thread will also load additional data into the shared memory
        if (global_index + blockDim.x < out_size + R - 1) {
            s_input[local_index + blockDim.x] = current_layer[global_index + blockDim.x];
        }
    }

    __syncthreads();

    // Forward pass
    if (global_index < out_size) {
        float sum = BIAS;
        for (int r = 0; r < R; r++) {
            sum += s_input[local_index + r] * W[base_index + N * r + threadIdx.x];
        }
        next_layer[global_index] = sigmoid(sum);
    }
}

__global__ void forwardpass(
    const float* __restrict__ current_layer,
    float* __restrict__ next_layer,
    const float* __restrict__ W,
    const int N,
    const int out_size
) {
    /* Computes the forward pass.
    
    Similar to the shared memory version, but without the first loading-part.

    @param current_layer Pointer to the input layer data (activations from the previous layer) in global memory.
    @param next_layer Pointer to the output layer data (activations for the current layer being computed) in global memory.
    @param W Pointer to the weights array in global memory.
    @param N Size (number of elements) of the first input layer (`current_layer`). Used for bounds checking during loading.
    @param out_size Size (number of elements) of the output layer (`next_layer`).
    */
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int base_index = blockIdx.x * blockDim.x;
    if (global_index < out_size) {
        float sum = BIAS;
        for (int r = 0; r < R; r++) {
            sum += current_layer[global_index + r] * W[base_index + N * r + threadIdx.x];
        }
        next_layer[global_index] = sigmoid(sum);
    }
}


int main(int argc, char** argv) {
    // Defining the input arguments
    int N = BLK_DIM;
    int K = 2;
    int machine_output = 1;
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (argc >= 4) machine_output = atoi(argv[3]);

    // Defining the metrics variables
    double t0, t1, t2, t3, throughput_shared, throughput_no_shared;

    // Allocating the "original" input layer and weight "matrix". Since we will check if
    // both kernels produces the same output, we need to load them using the same randomly
    // sampled data.
    float *original_input = (float*)malloc(N * sizeof(float));
    float *original_W = (float*)malloc(N * R * sizeof(float));

    // Define host pointer to the data
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_W = (float*)malloc(N * R * sizeof(float));
    float *results_1 = (float*)malloc(N * sizeof(float));
    float *results_2 = (float*)malloc(N * sizeof(float));

    // Filling the original arrays.
    for (int i = 0; i < N; i++) original_input[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < R * N; i++) original_W[i] = rand() / (float)RAND_MAX;

    // Create and allocate the necessary memory region on the device memory
    float *memory_region;
    cudaMalloc(&memory_region, (N + R*N + N) * sizeof(float));

    // Create pointer to the memory region, in order to access it correctly
    float *d_input = memory_region;
    float *d_W = (float*)((char*) memory_region + N * sizeof(float));
    float *d_output =  (float*)((char*) memory_region + (N + R*N) * sizeof(float));

    // Fill the host arrays
    for (int i = 0; i < N; i++) h_input[i] = original_input[i];
    for (int i = 0; i < R * N; i++) h_W[i] = original_W[i];

    // Copy the host array into the device memory.
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, N * R * sizeof(float), cudaMemcpyHostToDevice);

    t0 = hpc_gettime();
    // Forward pass for the shared kernel
    for (int i=0; i<K; i++) {
        int out_size = layer_size(N, i);
        int blocks = (out_size + BLK_DIM - 1) / BLK_DIM;
        forwardpass_shared<<<blocks, BLK_DIM>>>(d_input, d_output, d_W, N, out_size);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Layer swapping
        float* tmp = d_input; d_input = d_output; d_output = tmp;
    }
    t1 = hpc_gettime();

    // Metric computation
    throughput_shared = N / (t1-t0);

    // Copying results into the host pointer.
    cudaMemcpy(results_1, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Refilling the host variable with the original data and copying them into the device
    for (int i = 0; i < N; i++) h_input[i] = original_input[i];
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);


    t2 = hpc_gettime();
    // Forward pass without shared memory
    for (int i=0; i<K; i++) {
        int out_size = layer_size(N, i);
        int blocks = (out_size + BLK_DIM - 1) / BLK_DIM;
        forwardpass<<<blocks, BLK_DIM>>>(d_input, d_output, d_W, N, out_size);
        cudaDeviceSynchronize();
        cudaCheckError();

        // layer swapping
        float* tmp = d_input; d_input = d_output; d_output = tmp;
    }
    t3 = hpc_gettime();

    // Metric computation
    throughput_no_shared = N / (t3-t2);
    cudaMemcpy(results_2, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if the results are the same (Up to a threshold)
    bool flag = true;
    for (int i = 0; i < layer_size(N, K) and flag; i++) {
        if (abs(results_1[i] - results_2[i]) > 0.001) {
            flag = false;
        }
    }

    char output_control[30];
    if (flag) {
        strcpy(output_control , "Results are equal");
    }
    else {
        strcpy(output_control , "Results differ");
    }

    // Output the results

    if (machine_output)
        //Shared, thrShared, NoShared, thrNoShared, resultsequal
        printf("%f,%f,%f,%f,%i\n", t1-t0, throughput_shared, t3-t2, throughput_no_shared, flag);
    else {
        printf("N = {%i}, K = {%i}, R = {%i}\n", N, K, R);
        printf("Time taken for the no-shared memory kernel: %fs with a throughput of %f elements/second \n", t3 - t2, throughput_no_shared);
        printf("Time taken for the shared memory kernel: %fs with a throughput of %f elements/second \n", t1 - t0, throughput_shared);
        printf("%s between the two kernels!\n",output_control);
    }

    // Free the memory
    cudaFree(memory_region);
    free(h_input);
    free(h_W);
    cudaDeviceReset();

    return 0;
}
