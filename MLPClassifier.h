//
// Created by alery on 19/10/2021.
//

#ifndef CCPROJECT_MLPCLASSIFIER_H
#define CCPROJECT_MLPCLASSIFIER_H


#include "BaseMLP.h"

class MLPClassifier : public BaseMLP {
public:
    MLPClassifier(
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
    void PredictProba(int nbTests, const Matrix &inputs, Matrix &result) ;
};


#endif //CCPROJECT_MLPREGRESSOR_H
