cmake_minimum_required(VERSION 3.14)
project(ccproject)

set(CMAKE_CXX_STANDARD 11)

add_executable(ccproject main.cpp BaseMLP.cpp BaseMLP.h MLPRegressor.cpp MLPRegressor.h MLPClassifier.cpp MLPClassifier.h activations.h activations.cpp)