#ifndef NETWORK_H
#define NETWORK_H

typedef struct Node {
    double value;
} Node;

typedef struct Layer {
    Node* nodes;
    int size;
} Layer;

typedef struct Network {
    Layer* layers;
    int size;
    double** weight;
} Network;

Network initNetwork(int, int, int, int);
void freeNetwork(Network* network);
void printNetwork(Network network, int);

#endif //NETWORK_H
