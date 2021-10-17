#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

#include "MLP.h"

double fo(double);
double dfo(double);
double fh(double);
double dfh(double);

void readExample(double** inExamples, double** outExamples, int i, int nbInputs, int nbOutputs, const std::string &line);

int main(int argc, char ** argv) {

    if (argc != 7) {
        std::cerr << "use: ./ <data> <ratioExamplesTests> <nbHiddenUnits> <learningRate> <maxEpochs> <updatePeriod>" << std::endl;
        exit(1);
    }

    int nbExamples, nbInputs, nbOutputs;
    int nbHiddenUnits, maxEpochs, updatePeriod;
    double ratioExamplesTests, learningRate;

    ratioExamplesTests = atof(argv[2]);
    nbHiddenUnits = atoi(argv[3]);
    learningRate = atof(argv[4]);
    maxEpochs = atoi(argv[5]);
    updatePeriod = atoi(argv[6]);

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

    double **inTrainings = new double*[nbTrainings];
    double **outTrainings = new double*[nbTrainings];
    for (int i = 0; i < nbTrainings; i++) {
        if (!std::getline(infile, line)) {
            std::cerr << "Example " << i << " expected but not found." << std::endl;
            exit(0);
        }
        readExample(inTrainings, outTrainings, i, nbInputs, nbOutputs, line);
    }

    double **inTests = new double*[nbTests];
    double **outTests = new double*[nbTests];
    for (int i = 0; i < nbTests; i++) {
        if (!std::getline(infile, line)) {
            std::cerr << "Example " << i << " expected but not found." << std::endl;
            exit(0);
        }
        readExample(inTests, outTests, i, nbInputs, nbOutputs, line);
    }

    infile.close();

    MLP mlp(nbInputs, nbHiddenUnits, nbOutputs, fo, dfo, fh, dfh);
    mlp.train(inTrainings, outTrainings, nbTrainings, learningRate, maxEpochs, updatePeriod);
    mlp.test(inTests, outTests, nbTests);
    return 0;
}

void readExample(double** inExamples, double** outExamples, int i, int nbInputs, int nbOutputs, const std::string &line) {
    inExamples[i] = new double[nbInputs];
    outExamples[i] = new double[nbOutputs];

    std::istringstream iss(line);
    for (int j = 0; j < nbInputs; j++) {
        if (!(iss >> inExamples[i][j])) {
            std::cerr << "Expected input " << j << " for example " << i << std::endl;
            exit(0);
        }
    }

    for (int j = 0; j < nbOutputs; j++) {
        if (!(iss >> outExamples[i][j])) {
            std::cerr << "Expected output " << j << " for example " << i << std::endl;
            exit(0);
        }
    }
}

double fo(double x) {
    return 1 / (1 + exp(-x));
}

double dfo(double x) {
    return fo(x)*(1 - fo(x));
}

double fh(double x) {
    return 1 / (1 + exp(-x));
}

double dfh(double x) {
    return fh(x)*(1 - fh(x));
}
