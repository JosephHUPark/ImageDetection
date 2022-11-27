#pragma once

#include "neuron.hpp"
#include <vector>
#include <stdint.h>

static int layerID = 0;

class Layer
{
public:
    int currentLayerSize;
    std::vector<Neuron*> neurons;
    std::vector<double> layerOutput;
    Layer(int, int);
};