#include "utilities.h"

#include <math.h>
#include <stdlib.h>


double generate_random_double() {
    /* Generate values between -1 and 1*/
    return (double)(rand() % 1000000) / 500000 - 1;
}

double sigmoid(double x) {
    /* Sigmoid function */
    return 1.0 / (1.0 + exp(-x));
}
