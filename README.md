# COMP30230 - Connectionnist Computing - MLP

Aleryc SERRANIA - 21204068

## Requirements

- C++11 compiler
- CMake 3.14+

## Installation

With g++:

```sh
g++ -std=c++11 activations.cpp BaseMLP.cpp MLPClassifier.cpp MLPRegressor.cpp main.cpp -o mlp.exe
```

With cmake:

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config=Release
```

## Usage

```
mlp.exe <datafile> <mlpType> <ratioExamplesTests> <nbHiddenUnits> <hActivation> <oActivation> <learningRate> <maxEpochs> <updatePeriod> <printTestResult> <printFinalWeights>
    -datafile (str): input for training and test data
    -mlpType (r/c): r = regressor (output real values), c = classifier (output binary values)
    -ratioExamplesTests (real): real number to define the number of example to use in datafile
    -nbHiddenUnits (int): number of hidden units (1 layer only)
    -hActivation (str): activation function for hidden units (relu, sigmoid, tanh, linear)
    -oActivation (str): activation function for output units (relu, sigmoid, tanh, linear)
    -learningRate (real): the learning rate
    -maxEpochs (int): number of times the dataset is passed forward and backward in the network
    -updatePeriod (int): weights are updated every <updatePeriod> example
    -printTestResult (y/n): print the test result directly to stdout
    -printFinalWeights (y/n): print the weights at the end of the learning step to stdout
```

Example:

```sh
mlp.exe .\..\inputs\sin.input r 0.8 8 sigmoid linear 0.3 1000 1 n y
```
