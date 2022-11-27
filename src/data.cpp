#include "data.hpp"

Data::Data(){
    feature_vector = new std::vector<uint8_t>;
}

Data::~Data(){
    
}

void Data::set_feature_vector(std::vector<uint8_t> * vect){
    feature_vector = vect;
}
void Data::append_to_feature_vector(uint8_t val){
    feature_vector->push_back(val);
}
void Data::set_normalized_feature_vector(std::vector<double> * vect){
    normalized_feature_vector = vect;
}
void Data::append_to_normalized_feature_vector(double val){
    normalized_feature_vector->push_back(val);
}
void Data::set_class_vector(int count){
    class_vector = new std::vector<int>();
    for(int i = 0; i < count; i++){
        if(i == label)
            class_vector->push_back(1);
        else
            class_vector->push_back(0);
    }
}
void Data::set_label(uint8_t val){
    label = val;
}
void Data::set_enumerate_label(int val){
    enum_label = val;
}
void Data::set_distance(double val){
    distance = val;
}

int Data::get_feature_vector_size(){
    return feature_vector->size();
}
uint8_t Data::get_label(){
    return label;
}
uint8_t Data::get_enumerate_label(){
    return enum_label;
}

std::vector<uint8_t> * Data::get_feature_vector(){
    return feature_vector;
}
std::vector<double> * Data::get_normalized_feature_vector(){
    return normalized_feature_vector;
}
std::vector<int> * Data::get_class_vector(){
    return class_vector;
}
double Data::get_distance(){
    return distance;
}
