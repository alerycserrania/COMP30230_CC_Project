//
// Created by alery on 19/10/2021.
//

#include "MLPRegressor.h"

MLPRegressor::MLPRegressor(
        int nbInputs,
        int nbHiddens,
        int nbOutputs,
        double learningRate,
        int maxEpochs,
        int updatePeriod,
        const std::function<double(double)> &outputActivation,
        const std::function<double(double)> &dOutputActivation,
        const std::function<double(double)> &hiddenActivation,
        const std::function<double(double)> &dHiddenActivation
) : BaseMLP(nbInputs, nbHiddens, nbOutputs,
            learningRate, maxEpochs, updatePeriod,
            outputActivation, dOutputActivation,
            hiddenActivation, dHiddenActivation) {}

void MLPRegressor::Predict(int nbTests, const Matrix &inputs, Matrix &result) {
    Vector hiddens(nbHiddens), aHiddens(nbHiddens),
            outputs(nbOutputs), aOutputs(nbOutputs);
    result.resize(nbTests);
    for (int i = 0; i < nbTests; i++) {
        forward(inputs[i], hiddens, aHiddens, outputs, aOutputs);
        result[i].resize(nbOutputs);
        for (int j = 0; j < nbOutputs; j++) {
            result[i][j] = outputs[j];
        }
    }
}
