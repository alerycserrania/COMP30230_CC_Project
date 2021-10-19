//
// Created by alery on 19/10/2021.
//

#ifndef CCPROJECT_MLPREGRESSOR_H
#define CCPROJECT_MLPREGRESSOR_H


#include "BaseMLP.h"

class MLPRegressor : public BaseMLP {
public:
    MLPRegressor(
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
    );

    void Predict(int nbTests, const Matrix &inputs, Matrix &result) override;
};


#endif //CCPROJECT_MLPREGRESSOR_H
