#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <chrono>

#include "MLPRegressor.h"
#include "activations.h"
#include "MLPClassifier.h"


std::unordered_map<std::string, std::function<double(double)>> NAME_TO_ACTIVATION = {
        {"relu",    activation::relu},
        {"linear",  activation::linear},
        {"sigmoid", activation::sigmoid},
        {"tanh",    activation::hyperbolictan},
};

std::unordered_map<std::string, std::function<double(double)>> D_NAME_TO_ACTIVATION = {
        {"relu",    activation::drelu},
        {"linear",  activation::dlinear},
        {"sigmoid", activation::dsigmoid},
        {"tanh",    activation::dhyperbolictan},
};

void readExample(
        const std::string &line,
        Matrix &inExamples,
        Matrix &outExamples,
        int numExample,
        int nbInputs,
        int nbOutputs
);

int main(int argc, char **argv) {

    if (argc != 12) {
        std::cerr
                << "use: ./ <datafile> <mlpType> <ratioExamplesTests> <nbHiddenUnits> <hActivation> <oActivation> <learningRate> <maxEpochs> <updatePeriod> <printTestResult> <printFinalWeights>"
                << std::endl
                << "\t-datafile: input for training and test data" << std::endl
                << "\t-mlpType: r = regressor (output real values), c = classifier (output binary values)" << std::endl
                << "\t-ratioExamplesTests: real number to define the number of example to use in datafile" << std::endl
                << "\t-nbHiddenUnits: number of hidden units (1 layer only)" << std::endl
                << "\t-hActivation: activation function for hidden units (relu, sigmoid, tanh, linear)" << std::endl
                << "\t-oActivation: activation function for output units (relu, sigmoid, tanh, linear)" << std::endl
                << "\t-learningRate: the learning rate" << std::endl
                << "\t-maxEpochs: number of times the dataset is passed forward and backward in the network"
                << std::endl
                << "\t-updatePeriod: weights are updated every <updatePeriod> example" << std::endl
                << "\t-printTestResult (y/n): print the test result directly to stdout" << std::endl
                << "\t-printFinalWeights (y/n): print the weights at the end of the learning step to stdout"
                << std::endl;
        exit(1);
    }

    int nbExamples, nbInputs, nbOutputs;
    int nbHiddenUnits, maxEpochs, updatePeriod;
    double ratioExamplesTests, learningRate;
    std::string mlpType, hActivation, oActivation;
    char printTestResult, printFinalWeights;

    mlpType = argv[2];
    ratioExamplesTests = atof(argv[3]);
    nbHiddenUnits = atoi(argv[4]);
    hActivation = argv[5];
    oActivation = argv[6];
    learningRate = atof(argv[7]);
    maxEpochs = atoi(argv[8]);
    updatePeriod = atoi(argv[9]);
    printTestResult = argv[10][0];
    printFinalWeights = argv[11][0];

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

    std::cout << "type: " << mlpType << std::endl;
    std::cout << "nb trainings: " << nbTrainings << std::endl;
    std::cout << "nb tests: " << nbTests << std::endl;
    std::cout << "nb hidden units: " << nbHiddenUnits << std::endl;
    std::cout << "hidden activation: " << hActivation << std::endl;
    std::cout << "output activation: " << oActivation << std::endl;
    std::cout << "learning rate: " << learningRate << std::endl;
    std::cout << "max epochs: " << maxEpochs << std::endl;
    std::cout << "update period: " << updatePeriod << std::endl;
    std::cout << std::endl;

    if (mlpType == "r") {
        Matrix result;

        MLPRegressor mlp(
                nbInputs, nbHiddenUnits, nbOutputs,
                learningRate, maxEpochs, updatePeriod,
                NAME_TO_ACTIVATION[oActivation], D_NAME_TO_ACTIVATION[oActivation],
                NAME_TO_ACTIVATION[hActivation], D_NAME_TO_ACTIVATION[hActivation]
        );

        auto t_start = std::chrono::high_resolution_clock::now();
        mlp.Train(nbTrainings, inTrainings, outTrainings);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_diff = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << std::endl << "training time: " << t_diff;

        mlp.Predict(nbTests, inTests, result);

        double error = 0.0;
        for (int i = 0; i < nbTests; i++) {
            for (int j = 0; j < nbOutputs; j++) {
                error += pow(outTests[i][j] - result[i][j], 2);
            }
        }


        std::cout << std::endl;
        std::cout << "error cost: " << error / 2. << std::endl;

        if (printTestResult == 'y') {
            std::cout << std::endl;
            for (int i = 0; i < nbTests; i++) {
                std::cout << "got: ";
                for (int j = 0; j < nbOutputs; j++) {
                    std::cout << result[i][j] << ", ";
                }

                std::cout << "expected: ";
                for (int j = 0; j < nbOutputs; j++) {
                    std::cout << outTests[i][j] << ", ";
                }
                std::cout << std::endl;
            }
        }

        if (printFinalWeights == 'y') {
            std::cout << std::endl;
            std::cout << mlp << std::endl;
        }
    } else {
        Matrix result;

        MLPClassifier mlp(
                nbInputs, nbHiddenUnits, nbOutputs,
                learningRate, maxEpochs, updatePeriod,
                activation::sigmoid, activation::dsigmoid,
                activation::sigmoid, activation::dsigmoid
        );

        auto t_start = std::chrono::high_resolution_clock::now();
        mlp.Train(nbTrainings, inTrainings, outTrainings);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_diff = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << std::endl << "training time: " << t_diff;

        mlp.Predict(nbTests, inTests, result);

        int nbError = 0;
        for (int i = 0; i < nbTests; i++) {
            double prediction = result[i][0];
            double expected = std::distance(
                    outTests[i].cbegin(),
                    std::find(outTests[i].cbegin(), outTests[i].cend(), 1)
            );
            std::cout << prediction << ", " << expected << std::endl;
            if (prediction != expected) {
                nbError += 1;
            }


        }

        std::cout << std::endl;
        std::cout << "error rate: " << (double) nbError / (double) nbTests << std::endl;

        mlp.PredictProba(nbTests, inTests, result);

        double error = 0.0;
        for (int i = 0; i < nbTests; i++) {
            for (int j = 0; j < nbOutputs; j++) {
                error += pow(outTests[i][j] - result[i][j], 2);
            }
        }

        std::cout << "error cost: " << error / 2. << std::endl;

        if (printTestResult == 'y') {
            std::cout << std::endl;
            for (int i = 0; i < nbTests; i++) {
                std::cout << "got: ";
                for (int j = 0; j < nbOutputs; j++) {
                    std::cout << result[i][j] << ", ";
                }

                std::cout << "expected: ";
                for (int j = 0; j < nbOutputs; j++) {
                    std::cout << outTests[i][j] << ", ";
                }
                std::cout << std::endl;
            }
        }

        if (printFinalWeights == 'y') {
            std::cout << std::endl;
            std::cout << mlp << std::endl;
        }
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


