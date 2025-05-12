/****************************************************************************
*   OpenMP Multi Layer Neural Network Implementation.
*
*
*   On linux, compile using gcc with the following flags:
*        -std=c99 # Defines the C standard definition
*        -Wall # Set the warnings level to "all"
*        -Wpedantic # Enable extra warnings
*        -fopenmp # To enable OpenMP
*        -lm # To link the mathematical library
*    Example:
*        gcc -std=c99 -Wall -Wpedantic -fopenmp -Iinclude src/main.c src/network.c src/utilities.c -o openMP -lm
*
*    Or, if you have `cmake` and `make` avaiable
*        - path/to/root/folder$ mkdir build; cd build
*        - path/to/root/folder/build$ cmake ..
*        - path/to/root/folder/build$ make
*    This will create the executable.
*
*
*   Run the executable with:
*   ./openMP [N] [K] [machine output]
*
*   (N = first layer n° neurons)
*   (K = number of layers)
*   (machine output = 1 output machine readable, 0 output human readable. Default: 1)
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hpc.h"
#include "network.h"
#include "utilities.h"

/***************************
 * 
 *  CONSTANT DEFINITION
 * 
****************************/

// Define the "spread index" between layers. Each node is linked to the previous R nodes.
#define R 3

// Tells the program to run with a deterministic network, to check correctness.
#define deterministic 0


int main(int argc, char *argv[]) {
    // Setting the seed of the random number generator to have deterministic results
    srand(42);

    // Defining and reading input values
    int N = 100;
    int K = 2;
    int machine_output = 1;
    double b, thrput;
    if (!deterministic) {
        // run the program with random value and hyperparameters given in input
        if (argc >= 2) N = (int)strtol(argv[1], NULL, 10);
        if (argc >= 3) K = (int)strtol(argv[2], NULL, 10);
        if (argc >= 4 )machine_output = (int)strtol(argv[3], NULL, 10);
        b = generate_random_double();
    } else {
        // run the program with standard values and hyperparameters, to check correctness of the network
        N = 10;
        K = 3;
        b = 1.5;
    }
    
    // Check if the network is instantiatable
    if (N - K*(R - 1) < 1) {
        printf("Network is not instantiable! \n");
        return 1;
    }

    // Instantiate the network
    Network network = initNetwork(N, K, R, deterministic);

    // Forward pass
    const double begin = hpc_gettime();
    for (int i = 1; i < K; i++) {
        // Get the current layer size
        int current_layer_size = network.layers[i].size;
        // Split computation between threads
#pragma omp parallel for default(none) shared(current_layer_size, network, b, i)
        for (int j = 0; j < current_layer_size; j++) {
            // LOOP UNROLLING of the inner loop to speedup computation a bit (R is constant defined)
            double accumulator = network.layers[i-1].nodes[j].value * network.weight[j][0] +
                network.layers[i-1].nodes[j+1].value * network.weight[j][1] +
                network.layers[i-1].nodes[j+2].value * network.weight[j][2];
            // Assign the value to the node
            network.layers[i].nodes[j].value = sigmoid(accumulator + b);
        }
    }

    const double end = hpc_gettime();
    double elapsed_time = end - begin;

    // Compute metrics
    thrput = N / elapsed_time;

    if (machine_output) {
        printf("%f,%f,%i\n",elapsed_time,thrput,omp_get_max_threads());
    } else {
        printf("N = {%i}, K = {%i}, R = {%i}, OMP_NUM_THREAD = {%i}\n", N, K, R, omp_get_max_threads());
        printf("Time taken: %fs\n",elapsed_time);
    }
    

    //free the memory
    freeNetwork(&network);

    return 0;
}
