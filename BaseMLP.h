//
// Created by Aleryc Serrania on 17/10/2021.
//


#ifndef CCPROJECT_BASEMLP_H
#define CCPROJECT_BASEMLP_H

#include <functional>
#include <vector>


typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;

class BaseMLP {

public:
    BaseMLP(
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
    );

    void Train(
            int nbExamples,
            const Matrix &inputs,
            const Matrix &targets
    );

    virtual void Predict(
            int nbTests,
            const Matrix &inputs,
            Matrix &result
    ) = 0;

protected:
    void randomiseWeights();

    void forward(
            const Vector &inputs,
            Vector &hiddens,
            Vector &aHiddens,
            Vector &outputs,
            Vector &aOutputs
    );

    double backwards(
            const Vector &targets,
            const Vector &inputs,
            const Vector &hiddens,
            const Vector &aHiddens,
            const Vector &outputs,
            const Vector &aOutputs,
            Matrix &dwLower,
            Matrix &dwUpper
    );

    void updateWeights(
            Matrix &dwLower,
            Matrix &dwUpper
    );

protected:
    int nbInputs, nbHiddens, nbOutputs;
    Matrix wUpper, wLower;

    double learningRate;
    int maxEpochs, updatePeriod;

    std::function<double(double)> outputActivation, dOutputActivation;
    std::function<double(double)> hiddenActivation, dHiddenActivation;
};


#endif //CCPROJECT_BASEMLP_H
