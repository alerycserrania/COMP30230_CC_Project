//
// Created by alery on 19/10/2021.
//
#include <cmath>
#include "activations.h"


double activation::linear(double x) {
    return x;
}

double activation::dlinear(double x) {
    return 1;
}

double activation::relu(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

double activation::drelu(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

double activation::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double activation::dsigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double activation::hyperbolictan(double x) {
    return tanh(x);
}

double activation::dhyperbolictan(double x) {
    return pow(1 / cosh(x), 2);
}