#pragma once
#include "data.hpp"
#include <vector>

class common_data 
{
 protected:
 std::vector<Data *> *trainingData;
 std::vector<Data *> *testData;
 std::vector<Data *> *validationData;
 public:
 void setTrainingData(std::vector<Data *> * vect);
 void setTestData(std::vector<Data *> * vect);
 void setValidationData(std::vector<Data *> * vect);
};