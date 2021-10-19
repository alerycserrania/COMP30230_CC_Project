//
// Created by alery on 19/10/2021.
//
#include <algorithm>
#include <numeric>

#include "MLPClassifier.h"


MLPClassifier::MLPClassifier(
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

void MLPClassifier::Predict(int nbTests, const Matrix &inputs, Matrix &result) {
    Matrix probas;
    PredictProba(nbTests, inputs, probas);

    result.resize(nbTests);
    for (int i = 0; i < nbTests; i++) {
        auto highestProba(std::max_element(probas[i].cbegin(), probas[i].cend()));
        result[i].push_back(std::distance(probas[i].cbegin(), highestProba));
    }
}

void MLPClassifier::PredictProba(int nbTests, const Matrix &inputs, Matrix &result) {
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

    std::for_each(result.cbegin(), result.cend(), [](Vector v) -> void {
        double sum = std::accumulate(v.cbegin(), v.cend(), 0.0);
        std::transform(v.cbegin(), v.cend(), v.begin(), [sum](double d) -> double {
            return d / sum;
        });
    });
}
