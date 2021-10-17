//
// Created by Aleryc Serrania on 17/10/2021.
//

#include <iostream>

#include "MLP.h"

MLP::MLP(int nbInputs, int nbHiddens, int nbOutputs, double (*fo)(double), double (*dfo)(double), double (*fh)(double), double (*dfh)(double)) :
        nbInputs(nbInputs), nbHiddens(nbHiddens), nbOutputs(nbOutputs), fo(fo), dfo(dfo), fh(fh), dfh(dfh) {
    wUpper = new double*[nbOutputs];
    for (int j = 0; j < nbOutputs; j++) {
        wUpper[j] = new double[nbHiddens];
    }

    wLower = new double*[nbHiddens];
    for (int j = 0; j < nbHiddens; j++) {
        wLower[j] = new double[nbInputs];
    }

}

MLP::~MLP() {
    for (int j = 0; j < nbHiddens; j++) {
        delete[] wLower[j];
    }
    delete[] wLower;

    for (int j = 0; j < nbOutputs; j++) {
        delete[] wUpper[j];
    }
    delete[] wUpper;
}

void MLP::train(double **inputs, double **targets, int nbExamples, double learningRate, int maxEpochs, int updatePeriod) {
    double error;
    double *hiddens = new double[nbHiddens];
    double *outputs = new double[nbOutputs];
    double *aHiddens = new double[nbHiddens];
    double *aOutputs = new double[nbOutputs];
    double **dwLower = new double*[nbHiddens];
    double **dwUpper = new double*[nbOutputs];

    for (int j = 0; j < nbHiddens; j++) {
        dwLower[j] = new double[nbInputs];
        for (int i = 0; i < nbInputs; i++) {
            dwLower[j][i] = 0;
        }
    }

    for (int j = 0; j < nbOutputs; j++) {
        dwUpper[j] = new double[nbHiddens];
        for (int i = 0; i < nbHiddens; i++) {
            dwUpper[j][i] = 0;
        }
    }

    randomiseWeights();

    for (int e = 0; e < maxEpochs; e++) {
        error = 0;
        for (int p = 0; p < nbExamples; p++) {
            forward(inputs[p], hiddens, aHiddens, outputs, aOutputs);
            error += backwards(targets[p], inputs[p], hiddens, aHiddens, outputs, aOutputs, dwLower, dwUpper, learningRate);

            if (p % updatePeriod == 0) {
                updateWeights(dwLower, dwUpper, learningRate);
            }
        }
        error /= 2;
        std::cout << "Error at epoch " << e << " is " << error << "\n";
    }

    delete[] hiddens;
    delete[] outputs;
    delete[] aHiddens;
    delete[] aOutputs;

    for (int j = 0; j < nbHiddens; j++) {
        delete[] dwLower[j];
    }
    delete[] dwLower;

    for (int j = 0; j < nbOutputs; j++) {
        delete[] dwUpper[j];
    }
    delete[] dwUpper;
}

void MLP::forward(const double *inputs, double *hiddens, double *aHiddens, double *outputs, double *aOutputs) {
    for (int j = 0; j < nbHiddens; j++) {
        aHiddens[j] = 0;
        for (int i = 0; i < nbInputs; i++) {
            aHiddens[j] += wLower[j][i] * inputs[i];
        }
        hiddens[j] = fh(aHiddens[j]);
    }

    for (int j = 0; j < nbOutputs; j++) {
        aOutputs[j] = 0;
        for (int i = 0; i < nbHiddens; i++) {
            aOutputs[j] += wUpper[j][i] * hiddens[i];
        }
        outputs[j] = fo(aOutputs[j]);
    }
}

double MLP::backwards(const double *targets, const double *inputs, const double *hiddens, const double *aHiddens, const double *outputs, const double *aOutputs, double **dwLower, double **dwUpper, double learningRate) {
    double error = 0;
    double* oDelta = new double[nbOutputs];
    double* hDelta = new double[nbHiddens];

    // ok
    for (int j = 0; j < nbOutputs; j++) {
        oDelta[j] = (outputs[j] - targets[j])*dfo(aOutputs[j]);
        error += (targets[j] - outputs[j])*(targets[j] - outputs[j]);
    }

    // ok
    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens; i++) {
            dwUpper[j][i] += -learningRate*oDelta[j]*hiddens[i];
        }
    }

    for (int j = 0; j < nbHiddens; j++) {
        hDelta[j] = 0;
        for (int k = 0; k < nbOutputs; k++) {
            hDelta[j] += oDelta[k]*wUpper[k][j]*dfh(aHiddens[j]);
        }
    }

    // ok
    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs; i++) {
            dwLower[j][i] += -learningRate*hDelta[j]*inputs[i];
        }
    }

    delete[] oDelta;
    delete[] hDelta;

    return error;
}

void MLP::updateWeights(double **dwLower, double **dwUpper, double learningRate) {
    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs; i++) {
            wLower[j][i] += dwLower[j][i];
            dwLower[j][i] = 0;
        }
    }

    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens; i++) {
            wUpper[j][i] += dwUpper[j][i];
            dwUpper[j][i] = 0;
        }
    }
}

void MLP::test(double **inputs, double **targets, int nbTests) {
    double *hiddens = new double[nbHiddens];
    double *outputs = new double[nbOutputs];
    double *aHiddens = new double[nbHiddens];
    double *aOutputs = new double[nbOutputs];

    for (int i = 0; i < nbTests; i++) {
        forward(inputs[i], hiddens, aHiddens, outputs, aOutputs);
        std::cout << "(" << inputs[i][0] << ", " << inputs[i][1] << ") -> got: " << outputs[0] << ", expected: " << targets[i][0] << std::endl;
    }

    delete[] hiddens;
    delete[] outputs;
    delete[] aHiddens;
    delete[] aOutputs;
}

void MLP::randomiseWeights() {
    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens; i++) {
            wUpper[j][i] = ((((double)rand() / (double)RAND_MAX) * 2) - 1) / (nbOutputs * nbHiddens);
        }
    }

    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs; i++) {
            wLower[j][i] = ((((double)rand() / (double)RAND_MAX) * 2) - 1) / (nbHiddens * nbInputs);
        }
    }
}
