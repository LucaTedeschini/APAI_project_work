#include "utilities.h"
#include <math.h>
#include <stdlib.h>


double sigmoid(const double value) {
    return 1.0 / (1.0 + exp(-1.0 * value));
}

