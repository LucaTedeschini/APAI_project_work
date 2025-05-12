#include "network.h"

#include <stdio.h>

#include "utilities.h"
#include <stdlib.h>

Network initNetwork(int N, int K, int R, int b_deterministic) {
    /* Create the neural network. To keep a readable design, the network is basically an array of struct.
     * The network object itself is a struct, containing the size (which is K), the weight matrix and an array of layers
     * that contains their size and an array of nodes, that have a value.
     */
    Network network;

    if (!b_deterministic) {
        // Instantiate the network randomly with N and K given by the user
        int current_layer_size = 0;
        network.size = K;
        network.layers = malloc(network.size * sizeof(Layer));
        //allocate memory (and instantiate) nodes
        for (int i=0; i<K; i++) {
            current_layer_size = N - i*(R - 1);
            network.layers[i].nodes = malloc(current_layer_size * sizeof(Node));
            network.layers[i].size = current_layer_size;
            if (i == 0)
                for (int j=0; j<current_layer_size; j++) {
                    network.layers[i].nodes[j].value = generate_random_double();
                }
        }

        //allocate and instantiate weight matrix
        network.weight = malloc(N * sizeof(double*));
        for (int i=0; i<N; i++) {
            network.weight[i] = malloc(K * sizeof(double));
            for (int j=0; j<R; j++) {
                network.weight[i][j] = generate_random_double();
            }
        }
    } else {
        // Create the network with known weights and node's value
        int current_layer_size = 0;
        network.size = K;
        network.layers = malloc(network.size * sizeof(Layer));

        for (int i=0; i<K; i++) {
            current_layer_size = N - i*(R - 1);
            network.layers[i].nodes = malloc(current_layer_size * sizeof(Node));
            network.layers[i].size = current_layer_size;
            if (i == 0)
                for (int j=0; j<current_layer_size; j++) {
                    network.layers[i].nodes[j].value = 1.0 + j / 10.0;
                }
        }

        //allocate and instantiate weight matrix
        network.weight = malloc(N * sizeof(double*));
        for (int i=0; i<N; i++) {
            network.weight[i] = malloc(K * sizeof(double));
            for (int j=0; j<R; j++) {
                network.weight[i][j] = 0.5 + i / 10.0 + j / 10.0;
            }
        }

    }



    return network;
}


void freeNetwork(Network* network) {
    /* Free the network memory, freeing layer by layer */
    if (network == NULL) return;

    for (int i=0; i<network->layers->size; i++) {
        free(network->weight[i]);
    }
    free(network->weight);
    network->weight = NULL;

    for (int i = 0; i < network->size; i++) {
        free(network->layers[i].nodes);
        network->layers[i].nodes = NULL;
    }

    free(network->layers);
    network->layers = NULL;
    network->size = 0;
}

void printNetwork(Network network, int K) {
    for (int i = 0; i < K; i++) {
        printf("layer %i has %i nodes\n", i, network.layers[i].size);
        for (int j = 0; j < network.layers[i].size; j++) {
            printf("\tValue of %i node is: %f \n", j, network.layers[i].nodes[j].value);
        }
    }
}
