//function for adding in the model information
#ifndef ADD_MODEL_INFO_H
#define ADD_MODEL_INFO_H

#include <fstream>
#include <string>

using namespace std;

void add_model_info(int num_layers, int Ebedding_size, int LSTM_size, int target_vocab_size, int source_vocab_size, bool feed_input, string filename) {
    ifstream input(filename.c_str());
    string output_string = to_string(num_layers) + " " + to_string(Ebedding_size) + " " + to_string(LSTM_size) +" "+ to_string(target_vocab_size) +" "+ to_string(source_vocab_size) +" "+ to_string(feed_input);
    std::string str; 
    std::vector<std::string> file_lines;
    std::getline(input,str);//first line that is being replace
    file_lines.push_back(output_string);
    while(std::getline(input,str)) {
        file_lines.push_back(str);
    }
    input.close();
    std::ofstream output(filename.c_str());
    for(int i=0; i<file_lines.size(); i++) {
        output << file_lines[i] << "\n";
    }
    output.close();
} 
#endif
