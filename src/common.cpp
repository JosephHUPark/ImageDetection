#include "common.hpp"

void common_data::setTrainingData(std::vector<Data *> * vect)
{
 trainingData = vect;
}
void common_data::setTestData(std::vector<Data *> * vect)
{
 testData = vect;
}
void common_data::setValidationData(std::vector<Data *> * vect)
{
  validationData = vect;
}