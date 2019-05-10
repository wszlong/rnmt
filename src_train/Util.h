
#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#include <fstream>
//#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

using namespace std;

//counts the total number of words in my file format, so you can halve learning rate at half epochs
//counts the total number of lines too
void get_file_stats(int &num_lines,int &num_words,ifstream &input,int &total_target_words) {
	string str; 
	string word;
	num_lines =0;
	num_words=0;
	total_target_words=0;
    while (getline(input, str)){
        num_lines++;
    }

    input.clear();
	input.seekg(0, ios::beg);

    for(int i=0; i<num_lines; i+=4) {
    	getline(input, str);//source input
    	istringstream iss_input_source(str, istringstream::in);
    	while( iss_input_source >> word ) {
    		if(stoi(word) !=-1) {
    			num_words+=1;
    		}
    	}
    	getline(input, str);//source output,dont use
    	getline(input, str);//target input
    	istringstream iss_input_target(str, istringstream::in);
    	while( iss_input_target >> word ) {
    		if(stoi(word) != -1) {
    			num_words+=1;
    			total_target_words++;
    		}
    	}
    	getline(input, str);//target output,done use
    }
    input.clear();
	input.seekg(0, ios::beg);
}

/***
template<typename dType>
void write_matrix_GPU(dType *d_mat,int rows,int cols,std::ofstream &output) {
	
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	cudaMemcpy(temp_mat,d_mat,rows*cols*sizeof(dType),cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			output << temp_mat[IDX2C(i,j,rows)];
			if(j!=cols-1) {
				output << " ";
			}
		}
		output << "\n";
	}
	output << "\n";
	free(temp_mat);
}
***/

template<typename dType>
void write_matrix_GPU(dType *d_mat,int rows,int cols,std::ofstream &output) {
	
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	cudaMemcpy(temp_mat,d_mat,rows*cols*sizeof(dType),cudaMemcpyDeviceToHost);
	
	output.write((char *) temp_mat, rows*cols*sizeof(dType));
	free(temp_mat);
}

template<typename dType>
void read_matrix_GPU(dType *d_mat,int rows,int cols,std::ifstream &input) {

	vector<float> v;
	v.resize(rows*cols);
	input.read((char*)&v[0],sizeof(dType)*rows*cols);
	cudaMemcpy(d_mat,&v[0],rows*cols*sizeof(dType),cudaMemcpyHostToDevice);

}

/***
template<typename dType>
void read_matrix_GPU(dType *d_mat,int rows,int cols,std::ifstream &input) {

	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	
	std::string temp_string;
	std::string temp_token;
	for(int i=0; i<rows; i++) {
		//std::string temp_string;
		std::getline(input, temp_string);
		std::istringstream iss_input(temp_string, std::istringstream::in);
		for(int j=0; j<cols; j++) {
			//std::string temp_token;
			iss_input >> temp_token;
			//std::cout << temp_token << "\n";
			temp_mat[IDX2C(i,j,rows)] = std::stod(temp_token);
		}
	}
	std::getline(input, temp_string);

	cudaMemcpy(d_mat,temp_mat,rows*cols*sizeof(dType),cudaMemcpyHostToDevice);
	free(temp_mat);
}
***/

#endif
