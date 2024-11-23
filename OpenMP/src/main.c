#include <stdio.h>
#include <stdlib.h>
//#include <omp.h>
#include <time.h>
#include <hpc.h>
#include "utilities.h"



const float bias = 1.0f;



int main(int argc, char** argv){
    //Set the seed
    srand(42);
    // Read the parameters from cmdline
    const int N = (int)strtol(argv[1], NULL, 10);
    const int L = (int)strtol(argv[2], NULL, 10);
    const int R = (int)strtol(argv[3], NULL, 10);


    //Check if the network is instantiatable
    if (N - (L-1)*(R-1) < 1) {
        printf("ERROR: The network is not instantiatable\n");
        return 1;
    }


    // Creating the layer's value matrix
    double** values = malloc(L*sizeof(double*));

    // Compute the number of element necessary to store all the weights
    int total_weights = 0;
    for (int i = 0; i < L; i++) {
        total_weights += N - (i)*(R-1);
    }

    // Creating the weights matrix (as an array for easier access) Since all its element will be adjacent in memory
    double* w = malloc(total_weights*sizeof(double));


    // For each layer, assign the weights and instantiate the neurons' value
    int index = 0;
    for (int i=0; i<L; i++){
        const int neurons = N - (i)*(R-1);
        // Allocate the memory for the values, but do not instantiate it
        values[i] = (double*)malloc(neurons*sizeof(double));

        for (int j=0; j<neurons; j++){
            w[index + j] = -1 + ((double)rand() / RAND_MAX) * (2); //Gets value between -1 ad 1
            //Assigning values only to the first layer
            if (index == 0) values[0][j] = -1 + ((double)rand() / RAND_MAX) * (2); //Gets value between -1 ad 1
        }
        index += neurons;
    }


    // Forward pass
    const double begin = hpc_gettime();
    //I save the first layer in a support variable
    double* x = values[0];
    //Index is used to correctly access the weights array
    index = 0;
    for (int layer = 0; layer < L-1; layer++){
        // Compute the neurons for the next layer and for the current one
        const int neurons_next = N - (layer+1)*(R-1);
        const int neurons_current = N - (layer)*(R-1);
        // Create a pointer to the values of the next layer
        double* y = values[layer+1];
        
        #pragma omp parallel for default(none) shared(neurons_next, w, layer, x, bias, y, N, R, index)
        for (int j=0; j<neurons_next; j++){
            // Compute the value to assign
            double result = 0;
            for (int k=0; k<R; k++){
                result += w[index + (j+k)] * x[j];
            }
            // Assign the value
            y[j] = sigmoid(result + bias);
        }
        index += neurons_current;
        // make `x` point to the next layer. No need to free the memory here
        x = y;
    }
    const double end = hpc_gettime();
    const double time_spent = (double)(end - begin);
    printf("Took: %f\n",time_spent);

    //printf("Final neuron value: %f\n",values[2][0]);


    //Free the memory
    free(w);
    for (int i=0; i<L; i++){
        free(values[i]);
    }
    free(values);

    return 0;
}

// N = 5, L = 3, R = 3

/*
w[0] = 2.0;
w[1] = 4.0;
w[2] = 6.0;
w[3] = 8.0;
w[4] = 10.0;
w[5] = 0.5;
w[6] = 0.5;
w[7] = 0.5;
w[8] = 0.5;

values[0][0] = 0.5;
values[0][1] = 0.5;
values[0][2] = 0.5;
values[0][3] = 0.5;
values[0][4] = 0.5;

*/
// Final neuron value: 0.924046
