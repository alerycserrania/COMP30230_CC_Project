#include <iostream>
#include "MLP.h"

double fo(double);
double dfo(double);
double fh(double);
double dfh(double);

int main() {

    const double **inputs = new const double*[4];
    inputs[0] = new double[3] { 0, 0 };
    inputs[1] = new double[3] { 0, 1 };
    inputs[2] = new double[3] { 1, 0 };
    inputs[3] = new double[3] { 1, 1 };

    const double **targets = new const double*[4];
    targets[0] = new double[1] { 0 };
    targets[1] = new double[1] { 1 };
    targets[2] = new double[1] { 1 };
    targets[3] = new double[1] { 0 };


    MLP mlp(2, 5, 1, fo, dfo, fh, dfh);
    mlp.train(inputs, targets, 4, 0.5, 4000, 2);
    mlp.test(inputs, targets, 4);
    return 0;
}

double fo(double x) {
    return 1 / (1 + exp(-x));
}

double dfo(double x) {
    return fo(x)*(1 - fo(x));
}

double fh(double x) {
    return 1 / (1 + exp(-x));
}

double dfh(double x) {
    return fh(x)*(1 - fh(x));
}
