#pragma once

#include <vector>
#include "stdint.h"
#include "stdio.h"

class Data
{
    std::vector<uint8_t>* feature_vector;
    std::vector<double>* normalized_feature_vector;
    std::vector<int>* class_vector;
    uint8_t label;
    int enum_label;
    double distance;

    public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_normalized_feature_vector(std::vector<double> *);
    void append_to_normalized_feature_vector(double);
    void set_class_vector(int count);
    void set_label(uint8_t);
    void set_enumerate_label(int);
    void set_distance(double);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerate_label();

    std::vector<uint8_t> * get_feature_vector();
    std::vector<double> * get_normalized_feature_vector();
    std::vector<int> * get_class_vector();
    double get_distance();
};