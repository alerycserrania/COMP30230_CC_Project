//
// Created by Aleryc Serrania on 17/10/2021.
//

#ifndef CCPROJECT_MLP_H
#define CCPROJECT_MLP_H


class MLP {
public:
    MLP(int nbInputs, int nbHiddens, int nbOutputs, double (*fo)(double), double (*dfo)(double), double (*fh)(double), double (*dfh)(double));

    ~MLP();

    void train(double **inputs, double **targets, int nbExamples, double learningRate, int maxEpochs, int updatePeriod);

    void test(double **inputs, double **targets, int nbTests);

private:
    void randomiseWeights();

    void forward(const double *inputs, double *hiddens, double *aHiddens, double *outputs, double *aOutputs);
    double backwards(const double *targets, const double *inputs, const double *hiddens, const double *aHiddens, const double *outputs, const double *aOutputs, double **dwLower, double **dwUpper, double learningRate);
    void updateWeights(double** dwLower, double** dwUpper, double learningRate);
private:
    int nbInputs, nbHiddens, nbOutputs;
    double *bOutputs, *bHiddens;
    double **wUpper, **wLower;
    double (*fo)(double), (*dfo)(double);
    double (*fh)(double), (*dfh)(double);
};


#endif //CCPROJECT_MLP_H
