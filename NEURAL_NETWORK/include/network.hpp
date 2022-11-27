#pragma once

#include "../../include/data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "../../include/common.hpp"

class Network : public common_data
{
public:
    std::vector<Layer *> layers;
    double learningRate;
    double testPerformance;
    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> fprop(Data * data);
    double activate(std::vector<double>, std::vector<double>);
    double transfer(double);
    double transferDerivative(double);
    void bprop(Data * data);
    void updateWeights(Data * data);
    int predict(Data * data);
    void train(int);
    double test();
    void validate();
//     InputLayer *inputLayer;
//     OutputLayer *outputLayer;
//     std::vector<HiddenLayer *> hiddenLayers;
//     double eta; // learning rate

// public:
//     Network(std::vector<int> hiddenLayerSpec, int, int);
//     ~Network();
//     void fprop(Data * data);
//     void bprop(Data * data);
//     void updateWeights();
//     void train();
//     void test();
//     void validate();
};