
#ifndef FILE_INPUT
#define FILE_INPUT

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "Util.h"

using namespace std;

struct file_helper {
	string file_name; //Name of input file
	int minibatch_size; //Size of minibatches
	ifstream input_file; //Input file stream
	int current_line_in_file = 1;
	int nums_lines_in_file;
	
	//Used for computing the maximum sentence length of previous minibatch
	int words_in_minibatch;
	
	//-----------------------------------------GPU Parameters---------------------------------------------
	//This is for storing the vocab indicies on the GPU
	int max_sent_len; //max sentence length
	int current_source_length;
	int current_target_length;
	int source_vocab_size;
	int target_vocab_size;

	int *h_input_vocab_indicies_source;
	int *h_input_vocab_indicies_source_bi;
	int *h_output_vocab_indicies_source;

	int *h_input_vocab_indicies_target;
	int *h_output_vocab_indicies_target;

	int *h_input_vocab_indicies_source_temp;
	int *h_output_vocab_indicies_source_temp;
	int *h_input_vocab_indicies_target_temp;
	int *h_output_vocab_indicies_target_temp;
	
	//These are the special vocab indicies for the W gradient updates
	int *h_input_vocab_indicies_source_Wgrad;
	int *h_input_vocab_indicies_target_Wgrad;

	bool *bitmap_source; //This is for preprocessing the input vocab for quick updates on the W gradient
	bool *bitmap_target; //This is for preprocessing the input vocab for quick updates on the W gradient
	
	//length for the special W gradient stuff
	int len_source_Wgrad;
	int len_target_Wgrad;


	//for the attention model
	int *h_batch_info;

	//for perplexity
	int total_target_words;

	~file_helper() {

		delete [] bitmap_source;
		delete [] bitmap_target;
		
		free(h_input_vocab_indicies_source);
		free(h_input_vocab_indicies_source_bi);
		free(h_output_vocab_indicies_source);

		free(h_input_vocab_indicies_target);
		free(h_output_vocab_indicies_target);

		free(h_input_vocab_indicies_source_temp);
		free(h_output_vocab_indicies_source_temp);
		free(h_input_vocab_indicies_target_temp);
		free(h_output_vocab_indicies_target_temp);

		free(h_input_vocab_indicies_source_Wgrad);
		free(h_input_vocab_indicies_target_Wgrad);
		
		free(h_batch_info);

		input_file.close();
	}
	
	
	//can change to memset for speed if needed
	void zero_bitmaps() {

		for(int i=0; i<source_vocab_size; i++) {
			bitmap_source[i] = false;
		}

		for(int i=0; i<target_vocab_size; i++) {
			bitmap_target[i] = false;
		}
	}

	
	//This returns the length of the special sequence for the W grad
	void preprocess_input_Wgrad() {

		//zero out bitmaps at beginning
		zero_bitmaps();
		
		//For source
		for(int i=0; i<minibatch_size*current_source_length; i++) {

			if(h_input_vocab_indicies_source[i]==-1) {
				h_input_vocab_indicies_source_Wgrad[i] = -1;
			}
			else if(bitmap_source[h_input_vocab_indicies_source[i]]==false) {
				bitmap_source[h_input_vocab_indicies_source[i]]=true;
				h_input_vocab_indicies_source_Wgrad[i] = h_input_vocab_indicies_source[i];
			}
			else  {
				h_input_vocab_indicies_source_Wgrad[i] = -1;
			}
		}

		for(int i=0; i < minibatch_size*current_target_length; i++) {

			if(h_input_vocab_indicies_target[i]==-1) {
				h_input_vocab_indicies_target_Wgrad[i] = -1;
			}
			else if(bitmap_target[h_input_vocab_indicies_target[i]]==false) {
				bitmap_target[h_input_vocab_indicies_target[i]]=true;
				h_input_vocab_indicies_target_Wgrad[i] = h_input_vocab_indicies_target[i];
			}
			else  {
				h_input_vocab_indicies_target_Wgrad[i] = -1;
			}
		}

		//source
		//Now go and put all -1's at far right and number in far left
		// [13 -1 -1 9 -1 2 8 -1 -1] --> [13 8 2 9] 
		len_source_Wgrad = -1;
		int left_index = 0;
		int right_index = minibatch_size*current_source_length-1;
		while(left_index < right_index) {
			if(h_input_vocab_indicies_source_Wgrad[left_index]==-1) {
				if(h_input_vocab_indicies_source_Wgrad[right_index]!=-1) {
					int temp_swap = h_input_vocab_indicies_source_Wgrad[left_index];
					h_input_vocab_indicies_source_Wgrad[left_index] = h_input_vocab_indicies_source_Wgrad[right_index];
					h_input_vocab_indicies_source_Wgrad[right_index] = temp_swap;
					left_index++;
					right_index--;
					continue;
				}
				else {
					right_index--;
					continue;
				}
			}
			left_index++;
		}
		if(h_input_vocab_indicies_source_Wgrad[left_index]!=-1) {
			left_index++;
		}
		len_source_Wgrad = left_index;

		//target
		//Now go and put all -1's at far right and number in far left
		len_target_Wgrad = -1;
		left_index = 0;
		right_index = minibatch_size*current_target_length-1;
		while(left_index < right_index) {
			if(h_input_vocab_indicies_target_Wgrad[left_index]==-1) {
				if(h_input_vocab_indicies_target_Wgrad[right_index]!=-1) {
					int temp_swap = h_input_vocab_indicies_target_Wgrad[left_index];
					h_input_vocab_indicies_target_Wgrad[left_index] = h_input_vocab_indicies_target_Wgrad[right_index];
					h_input_vocab_indicies_target_Wgrad[right_index] = temp_swap;
					left_index++;
					right_index--;
					continue;
				}
				else {
					right_index--;
					continue;
				}
			}
			left_index++;
		}
		if(h_input_vocab_indicies_target_Wgrad[left_index]!=-1) {
			left_index++;
		}
		len_target_Wgrad = left_index;
	}


	//Constructor
	file_helper(string fn,int ms,int &nlif,int max_sent_len,int source_vocab_size,int target_vocab_size,int &total_words)
	{
		file_name = fn;
		minibatch_size = ms;
		input_file.open(file_name.c_str(),ifstream::in); //Open the stream to the file
		this->source_vocab_size = source_vocab_size;
		this->target_vocab_size = target_vocab_size;

		get_file_stats(nlif,total_words,input_file,total_target_words);
		nums_lines_in_file = nlif;

		//GPU allocation
		this->max_sent_len = max_sent_len;
		h_input_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_input_vocab_indicies_source_bi = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_target = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_target = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_input_vocab_indicies_target_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_target_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_Wgrad = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_input_vocab_indicies_target_Wgrad = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		bitmap_source = new bool[source_vocab_size*sizeof(bool)];
		bitmap_target = new bool[target_vocab_size*sizeof(bool)];
		
		h_batch_info = (int *)malloc(2*minibatch_size * sizeof(int));
		
	}

	//Read in the next minibatch from the file
	//returns bool, true is same epoch, false if now need to start new epoch
	bool read_minibatch() {

		bool sameEpoch = true;
		words_in_minibatch=0; //For throughput calculation

		//For gpu file input
		int current_temp_source_input_index = 0;
		int current_temp_source_output_index = 0;
		int current_temp_target_input_index = 0;
		int current_temp_target_output_index = 0;

		//std::cout << "Begin minibatch(Now printing input that was in the file)\n";
		//Now load in the minibatch
		//std::cout << "current_line_in_file: " << current_line_in_file << "\n";
		//std::cout << "nums_lines_in_file: " << nums_lines_in_file << "\n";
		for(int i=0; i<minibatch_size; i++) {
			if(current_line_in_file > nums_lines_in_file) {
				input_file.clear();
				input_file.seekg(0, ios::beg);
				current_line_in_file = 1;
				sameEpoch = false;

				break;
			}

			string temp_input_source;
			string temp_output_source;
			getline(input_file, temp_input_source);
			getline(input_file, temp_output_source);

			string temp_input_target;
			string temp_output_target;
			getline(input_file, temp_input_target);
			getline(input_file, temp_output_target);

			///////////////////////////////////Process the source////////////////////////////////////
			istringstream iss_input_source(temp_input_source, istringstream::in);
			istringstream iss_output_source(temp_output_source, istringstream::in);
			string word; //The temp word

			int input_source_length = 0;
			while( iss_input_source >> word ) {
				//std::cout << word << " ";
				h_input_vocab_indicies_source_temp[current_temp_source_input_index] = stoi(word);
				input_source_length+=1;
				current_temp_source_input_index+=1;
			}
			//std::cout << "\n";
			int output_source_length = 0;
			while( iss_output_source >> word ) {
				//std::cout << word << " ";
				h_output_vocab_indicies_source_temp[current_temp_source_output_index] = stoi(word);
				output_source_length+=1;
				current_temp_source_output_index+=1;
			}

			words_in_minibatch+=input_source_length;
			//max_sent_len_source = input_source_length;

			///////////////////////////////////Process the target////////////////////////////////////
			istringstream iss_input_target(temp_input_target, istringstream::in);
			istringstream iss_output_target(temp_output_target, istringstream::in);

			int input_target_length = 0;
			while( iss_input_target >> word ) {
				//std::cout << word << " ";
				h_input_vocab_indicies_target_temp[current_temp_target_input_index] = stoi(word);
				current_temp_target_input_index+=1;
				input_target_length+=1;
			}
			//std::cout << "\n";
			int output_target_length = 0;
			while( iss_output_target >> word ) {
				//std::cout << word << " ";
				h_output_vocab_indicies_target_temp[current_temp_target_output_index] = stoi(word);
				current_temp_target_output_index+=1;
				output_target_length+=1;
			}

			current_source_length = input_source_length;
			current_target_length = input_target_length;
			words_in_minibatch += input_target_length; 
			//max_sent_len_target = input_target_length;

			//Now increase current line in file because we have seen two more sentences
			current_line_in_file+=4;
		}

		if(current_line_in_file>nums_lines_in_file) {
			current_line_in_file = 1;
			input_file.clear();
			input_file.seekg(0, ios::beg);
			sameEpoch = false;
		}

		//reset for GPU
		words_in_minibatch = 0;

		//get vocab indicies in correct memory layout on the host
		//std::cout << "-------------------source input check--------------------\n";
		for(int i=0; i<minibatch_size; i++) {
			int STATS_source_len = 0;
			for(int j=0; j<current_source_length; j++) {

				//stuff for getting the individual source lengths in the minibatch
				if(h_input_vocab_indicies_source_temp[j + current_source_length*i]!=-1) {
					STATS_source_len+=1;
				}
				h_input_vocab_indicies_source[i + j*minibatch_size] = h_input_vocab_indicies_source_temp[j + current_source_length*i];
				h_input_vocab_indicies_source_bi[i + (current_source_length-1-j)*minibatch_size] = h_input_vocab_indicies_source[i + j*minibatch_size];
				
				h_output_vocab_indicies_source[i + j*minibatch_size] = h_output_vocab_indicies_source_temp[j + current_source_length*i];
				if(h_input_vocab_indicies_source[i + j*minibatch_size]!=-1) {
					words_in_minibatch+=1;
				}
			}
			h_batch_info[i] = STATS_source_len;
			h_batch_info[i+minibatch_size] = current_source_length - STATS_source_len;
		}


		//std::cout << "-------------------target input check--------------------\n";
		for(int i=0; i<minibatch_size; i++) {
			for(int j=0; j<current_target_length; j++) {
				//std::cout << "i, j, current target length: " << i << " " << j << " " << current_target_length << "\n";
				h_input_vocab_indicies_target[i + j*minibatch_size] = h_input_vocab_indicies_target_temp[j + current_target_length*i];
				h_output_vocab_indicies_target[i + j*minibatch_size] = h_output_vocab_indicies_target_temp[j + current_target_length*i];
				if(h_output_vocab_indicies_target[i + j*minibatch_size]!=-1) {
					words_in_minibatch+=1;
				}
			}
		}

		//Now preprocess the data on the host before sending it to the gpu
		//for updating word embedding quickly
		preprocess_input_Wgrad(); 

		return sameEpoch;
	}
};

#endif
	
