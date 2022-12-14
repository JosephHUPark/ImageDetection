#include "data_handler.hpp"

data_handler::data_handler(){
    data_array = new std::vector<Data*>;
    training_data = new std::vector<Data*>;
    test_data = new std::vector<Data*>;
    validation_data = new std::vector<Data*>; 
}
data_handler::~data_handler(){
    
}

void data_handler::normalize(){
    std::vector<double> mins, maxs;
    
    Data * d = data_array->at(0);
    for(auto val: *d->get_feature_vector()){
        mins.push_back(val);
        maxs.push_back(val);
    }

    for(int i = 0; i < data_array->size(); i++)
    {
        d = data_array->at(i);
        for(int j = 0; j < d->get_feature_vector_size(); j++){
            double value = (double) d->get_feature_vector()->at(j);
            if(value < mins.at(j)) mins[j] = value;
            if(value < maxs.at(j)) maxs[j] = value;
        }
    }

    for(int i = 0; i < data_array->size(); i++){
        data_array->at(i)->set_normalized_feature_vector(new std::vector<double>);
        data_array->at(i)->set_class_vector(num_classes);
        for(int j = 0; j < data_array->at(i)->get_feature_vector_size(); j++){
            if(maxs[j] - mins[j] == 0) data_array->at(i)->append_to_feature_vector(0.0);
            else
                data_array->at(i)->append_to_feature_vector((double)(data_array->at(i)->get_feature_vector()->at(j) - mins[j]/(maxs[j]-mins[j])));
        }
    }
}
void data_handler::read_csv(std::string path, std::string delimiter){
    num_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line;
    while(std::getline(data_file, line)){
        if(line.length() == 0) continue;
        Data* d = new Data();
        d->set_normalized_feature_vector(new std::vector<double>());
        size_t position = 0;
        std::string token;
        while((position = line.find(delimiter)) != std::string::npos){
            token = line.substr(0, position);
            d->append_to_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        if(classMap.find(line) != classMap.end()){
            d->set_label(classMap[line]);
        }
        else{
            classMap[line] = num_classes;
            d->set_label(classMap[line]);
            num_classes++;
        }
        data_array->push_back(d);
    }
    feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
}
void data_handler::read_feature_vector(std::string path){
    uint32_t header[4];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if(f)
    {
        for(int i = 0; i < 4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }


        printf("%lu   %lu   %lu   %lu\n", header[0], header[1], header[2], header[3]);
        printf("Finished getting file header\n");
        int image_size = header[2]*header[3];
        for(int i = 0; i < header[1]; i++)
        {
            Data *d = new Data();
            uint8_t element[1];
            for(int j = 0; j < image_size; j++)
            {
                if(fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    printf("Error reading from File.\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
    }
    else{
        printf("Could not find file.\n");
        exit(1);
    }
}
void data_handler::read_feature_labels(std::string path){
    uint16_t header[2];
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if(f)
    {
        for(int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }

        printf("%lu   %lu\n", header[0], header[1]);
        printf("Finished getting Label File Header\n");
        for(int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                printf("Error reading from File.\n");
                exit(1); 
            }
        }
        printf("Successfully read and stored label.\n");
    }
    else{
        printf("Could not find file.\n");
        exit(1);
    }
}
void data_handler::split_data(){
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_PERCENT;

    printf("Data Array size: %d\n", data_array->size());
    printf("Training size: %d\n", train_size);
    printf("Test size: %d\n", test_size);
    printf("Valid size: %d\n", valid_size);

    // Train Data
    int count = 0;
    while(count < train_size)   
    {
        int big = rand()<<16;
        int rand_index = (big | rand())%data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end()){
            training_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }
    printf("Training data size: %lu\n", training_data->size());

    // Test Data
    count = 0;
    while(count < test_size)
    {
        int big = rand()<<16;
        int rand_index = (big | rand())%data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end()){
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }
    printf("Test data size: %lu\n", test_data->size());

    // Validation Data
    count = 0;
    while(count < valid_size)
    {
        int big = rand()<<16;
        int rand_index = (big | rand())%data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end()){
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }
    printf("Validation data size: %lu\n", validation_data->size());
}
void data_handler::count_classes(){
    int count = 0;
    for(unsigned int i = 0; i < data_array->size(); i++){
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end()){
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerate_label(count);
            count++;
        }
    }
    num_classes = count;
    for(Data *data: *data_array){
        data->set_class_vector(num_classes);
    }
    printf("Succesfully extracted %d classes.\n", num_classes); 
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes){
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

int data_handler::get_class_counts(){
    return num_classes;
}
std::vector<Data*>* data_handler::get_training_data(){
    return training_data;
}
std::vector<Data*>* data_handler::get_test_data(){
    return test_data;
}
std::vector<Data*>* data_handler::get_validation_data(){
    return validation_data;
}

// int main(){
//     data_handler *dh = new data_handler();
//     dh->read_feature_vector("./train-images.idx3-ubyte");
//     dh->read_feature_labels("./train-labels.idx1-ubyte");
//     dh->split_data();
//     dh->count_classes();
// }