#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

#include "MLPRegressor.h"
#include "activations.h"
#include "MLPClassifier.h"


void readExample(
        const std::string &line,
        Matrix &inExamples,
        Matrix &outExamples,
        int numExample,
        int nbInputs,
        int nbOutputs
);

int main(int argc, char **argv) {

    if (argc != 8) {
        std::cerr
                << "use: ./ <datafile> <mlpType=r|c> <ratioExamplesTests> <nbHiddenUnits> <learningRate> <maxEpochs> <updatePeriod>"
                << std::endl;
        exit(1);
    }

    int nbExamples, nbInputs, nbOutputs;
    int nbHiddenUnits, maxEpochs, updatePeriod;
    double ratioExamplesTests, learningRate;
    std::string mlpType;

    mlpType = argv[2];
    ratioExamplesTests = atof(argv[3]);
    nbHiddenUnits = atoi(argv[4]);
    learningRate = atof(argv[5]);
    maxEpochs = atoi(argv[6]);
    updatePeriod = atoi(argv[7]);

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Could not open file '" << argv[1] << "'" << std::endl;
        exit(0);
    }

    std::string line;
    if (!std::getline(infile, line)) {
        std::cerr << "Missing nbExamples, nbInputs and nbOutputs in data file" << std::endl;
        exit(0);
    }

    std::istringstream iss(line);
    if (!(iss >> nbExamples >> nbInputs >> nbOutputs)) {
        std::cerr << "Missing nbExamples, nbInputs and nbOutputs in data file" << std::endl;
        exit(0);
    }

    int nbTrainings = int(nbExamples * ratioExamplesTests);
    int nbTests = nbExamples - nbTrainings;

    Matrix inTrainings(nbTrainings);
    Matrix outTrainings(nbTrainings);
    for (int i = 0; i < nbTrainings; i++) {
        if (!std::getline(infile, line)) {
            std::cerr << "Example " << i << " expected but not found." << std::endl;
            exit(0);
        }
        readExample(line, inTrainings, outTrainings, i, nbInputs, nbOutputs);
    }

    Matrix inTests(nbTests);
    Matrix outTests(nbTests);
    for (int i = 0; i < nbTests; i++) {
        if (!std::getline(infile, line)) {
            std::cerr << "Example " << i << " expected but not found." << std::endl;
            exit(0);
        }
        readExample(line, inTests, outTests, i, nbInputs, nbOutputs);
    }

    infile.close();

    if (mlpType == "r") {
        Matrix result;

        MLPRegressor mlp(
                nbInputs, nbHiddenUnits, nbOutputs,
                learningRate, maxEpochs, updatePeriod,
                activation::sigmoid, activation::dsigmoid,
                activation::sigmoid, activation::dsigmoid
        );
        mlp.Train(nbTrainings, inTrainings, outTrainings);
        mlp.Predict(nbTests, inTests, result);

        double error = 0.0;
        for (int i = 0; i < nbTests; i++) {
            for (int j = 0; j < nbOutputs; j++) {
                error += pow(outTests[i][j] - result[i][j], 2);
            }

            std::cout << "Expected: ";
            for (int j = 0; j < nbOutputs; j++) {
                std::cout << outTests[i][j] << " ";
            }
            std::cout << ", Predicted: ";
            for (int j = 0; j < nbOutputs; j++) {
                std::cout << result[i][j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Error: " << error / 2. << std::endl;
    } else {
        Matrix result;

        MLPClassifier mlp(
                nbInputs, nbHiddenUnits, nbOutputs,
                learningRate, maxEpochs, updatePeriod,
                activation::sigmoid, activation::dsigmoid,
                activation::sigmoid, activation::dsigmoid
        );
        mlp.Train(nbTrainings, inTrainings, outTrainings);
        mlp.Predict(nbTests, inTests, result);

        int nbError = 0;
        for (int i = 0; i < nbTests; i++) {
            double prediction = result[i][0];
            double expected = std::distance(
                    outTests[i].cbegin(),
                    std::find(outTests[i].cbegin(), outTests[i].cend(), 1)
            );

            std::cout << "Prediction: " << result[i][0];
            std::cout << ", Expected: " << expected << std::endl;

            if (prediction != expected) {
                nbError += 1;
            }
        }
        double errorRate = (double) nbError / (double) nbTests;
        std::cout << "Error rate: " << errorRate << "(" << nbError << "/" << nbTests << ")" << std::endl;
    }

    return 0;
}

void readExample(
        const std::string &line,
        Matrix &inExamples,
        Matrix &outExamples,
        int numExample,
        int nbInputs,
        int nbOutputs
) {
    inExamples[numExample].resize(nbInputs);
    outExamples[numExample].resize(nbOutputs);

    std::istringstream iss(line);
    for (int j = 0; j < nbInputs; j++) {
        if (!(iss >> inExamples[numExample][j])) {
            std::cerr << "Expected input " << j << " for example " << numExample << std::endl;
            exit(0);
        }
    }

    for (int j = 0; j < nbOutputs; j++) {
        if (!(iss >> outExamples[numExample][j])) {
            std::cerr << "Expected output " << j << " for example " << numExample << std::endl;
            exit(0);
        }
    }
}


