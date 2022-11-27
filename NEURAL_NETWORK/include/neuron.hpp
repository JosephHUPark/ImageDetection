#pragma once

#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron{
public:
    double output;
    double delta;
    std::vector<double> weights;
    Neuron(int, int);
    void initializeWeights(int);
//     std::vector<double> weights;
//     double preActivation;
//     double activatedOutput;
//     double outputDerivative;
//     double error;
//     double alpha;

// public:
//     Neuron(int, int);
//     ~Neuron();
//     void initializeWeights(int previousLayerSize, int currentLayerSize);
//     void setError(double);
//     void setWeight(double, int);
//     void calculatePreActivation(std::vector<double>);
//     double activate();
//     double calculateOutputDerivative();
//     double sigmoid();
//     double relu();
//     double leakyRelu();
//     double inverseSqrtRelu();
//     double getOutput();
//     double getOutputDerivative();
//     double getError();
//     std::vector<double> getWeights();
};