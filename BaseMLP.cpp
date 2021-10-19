//
// Created by Aleryc Serrania on 17/10/2021.
//

#include <iostream>
#include <functional>
#include <utility>

#include "BaseMLP.h"


BaseMLP::BaseMLP(
        int nbInputs,
        int nbHiddens,
        int nbOutputs,
        double learningRate,
        int maxEpochs,
        int updatePeriod,
        std::function<double(double)> outputActivation,
        std::function<double(double)> dOutputActivation,
        std::function<double(double)> hiddenActivation,
        std::function<double(double)> dHiddenActivation
) :
        nbInputs(nbInputs),
        nbHiddens(nbHiddens),
        nbOutputs(nbOutputs),
        learningRate(learningRate),
        maxEpochs(maxEpochs),
        updatePeriod(updatePeriod),
        outputActivation(std::move(outputActivation)),
        dOutputActivation(std::move(dOutputActivation)),
        hiddenActivation(std::move(hiddenActivation)),
        dHiddenActivation(std::move(dHiddenActivation)),
        wUpper(nbOutputs),
        wLower(nbHiddens) {

    for (int j = 0; j < nbHiddens; j++) {
        wLower[j].resize(nbInputs + 1);
    }

    for (int j = 0; j < nbOutputs; j++) {
        wUpper[j].resize(nbHiddens + 1);
    }
}

void BaseMLP::Train(int nbExamples, const Matrix &inputs, const Matrix &targets) {
    Vector hiddens(nbHiddens), aHiddens(nbHiddens),
            outputs(nbOutputs), aOutputs(nbOutputs);
    Matrix dwLower(nbHiddens), dwUpper(nbOutputs);
    double error;

    for (int j = 0; j < nbHiddens; j++) {
        dwLower[j].resize(nbInputs + 1);
    }

    for (int j = 0; j < nbOutputs; j++) {
        dwUpper[j].resize(nbHiddens + 1);
    }

    randomiseWeights();

    for (int e = 0; e < maxEpochs; e++) {
        error = 0;
        for (int p = 0; p < nbExamples; p++) {
            forward(inputs[p], hiddens, aHiddens, outputs, aOutputs);
            error += backwards(targets[p], inputs[p], hiddens, aHiddens, outputs, aOutputs, dwLower, dwUpper);

            if (p % updatePeriod == 0) {
                updateWeights(dwLower, dwUpper);
            }
        }
        error /= 2;
        if (e % 10 == 0) std::cout << "Error at epoch " << e << " is " << error << "\n";
    }
}



void BaseMLP::forward(const Vector &inputs, Vector &hiddens, Vector &aHiddens, Vector &outputs, Vector &aOutputs) {
    for (int j = 0; j < nbHiddens; j++) {
        aHiddens[j] = 0;
        for (int i = 0; i < nbInputs; i++) {
            aHiddens[j] += wLower[j][i] * inputs[i];
        }
        aHiddens[j] += wLower[j][nbInputs];
        hiddens[j] = hiddenActivation(aHiddens[j]);
    }

    for (int j = 0; j < nbOutputs; j++) {
        aOutputs[j] = 0;
        for (int i = 0; i < nbHiddens; i++) {
            aOutputs[j] += wUpper[j][i] * hiddens[i];
        }
        aOutputs[j] += wUpper[j][nbHiddens];
        outputs[j] = outputActivation(aOutputs[j]);
    }
}

double BaseMLP::backwards(
        const Vector &targets,
        const Vector &inputs,
        const Vector &hiddens,
        const Vector &aHiddens,
        const Vector &outputs,
        const Vector &aOutputs,
        Matrix &dwLower,
        Matrix &dwUpper
) {
    double error(0);
    Vector oDelta(nbOutputs), hDelta(nbHiddens);

    for (int j = 0; j < nbOutputs; j++) {
        oDelta[j] = (outputs[j] - targets[j]) * dOutputActivation(aOutputs[j]);
        error += (targets[j] - outputs[j]) * (targets[j] - outputs[j]);
    }

    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens; i++) {
            dwUpper[j][i] += -learningRate * oDelta[j] * hiddens[i];
        }
        dwUpper[j][nbHiddens] += -learningRate * oDelta[j];
    }

    for (int j = 0; j < nbHiddens; j++) {
        hDelta[j] = 0;
        for (int k = 0; k < nbOutputs; k++) {
            hDelta[j] += oDelta[k] * wUpper[k][j] * dHiddenActivation(aHiddens[j]);
        }
    }

    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs; i++) {
            dwLower[j][i] += -learningRate * hDelta[j] * inputs[i];
        }
        dwLower[j][nbInputs] += -learningRate * hDelta[j];
    }

    return error;
}

void BaseMLP::updateWeights(Matrix &dwLower, Matrix &dwUpper) {
    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs + 1; i++) {
            wLower[j][i] += dwLower[j][i];
            dwLower[j][i] = 0;
        }
    }

    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens + 1; i++) {
            wUpper[j][i] += dwUpper[j][i];
            dwUpper[j][i] = 0;
        }
    }
}

void BaseMLP::randomiseWeights() {
    for (int j = 0; j < nbOutputs; j++) {
        for (int i = 0; i < nbHiddens; i++) {
            wUpper[j][i] = ((((double) rand() / (double) RAND_MAX) * 2) - 1) / (nbOutputs * nbHiddens);
        }
    }

    for (int j = 0; j < nbHiddens; j++) {
        for (int i = 0; i < nbInputs; i++) {
            wLower[j][i] = ((((double) rand() / (double) RAND_MAX) * 2) - 1) / (nbHiddens * nbInputs);
        }
    }
}
