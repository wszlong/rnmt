
#ifndef INPUT_FILE_PREP_H
#define INPUT_FILE_PREP_H

#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include "CUDA_UTIL.h"

using namespace std;

struct comb_sent_info {

	std::vector<std::string> src_sent;
	std::vector<std::string> tgt_sent;

	std::vector<int> src_sent_int;
	std::vector<int> minus_two_source;
	std::vector<int> tgt_sent_int_i;
	std::vector<int> tgt_sent_int_o;
	int total_len;

	comb_sent_info(std::vector<std::string> &src_sent,std::vector<std::string> &tgt_sent) {
		this->src_sent = src_sent;
		this->tgt_sent = tgt_sent;
		total_len = tgt_sent.size() + src_sent.size();
	}
};

struct compare_nonLM {
    bool operator()(const struct comb_sent_info& first, const struct comb_sent_info& second) {
        return first.total_len < second.total_len;
    }
};

struct mapping_pair {
	std::string word;
	int count;
	mapping_pair(std::string word,int count) {
		this->word = word;
		this->count = count;
	}
};

struct mapping_pair_compare_functor {
  	bool operator() (mapping_pair &a,mapping_pair &b) const { return (a.count < b.count); }
};


struct input_file_prep {

	std::ifstream source_input;
	std::ifstream target_input;
	std::ofstream final_output;

	std::unordered_map<std::string,int> src_mapping;
	std::unordered_map<std::string,int> tgt_mapping;

	std::unordered_map<std::string,int> src_counts;
	std::unordered_map<std::string,int> tgt_counts;

	const int minibatch_mult = 10;
	vector<comb_sent_info> data;
	
	bool prep_files_train_nmt(int minibatch_size, int max_sent_cutoff, 
		string source_file_name, string target_file_name, string output_file_name,
		int &source_vocab_size, int &target_vocab_size, bool shuffle, 
		string model_output_file_name, int embedding_size, int hiddenstate_size, int num_layers)
	{
		int VISUAL_num_source_word_tokens = 0;
		int VISUAL_total_source_vocab_size = 0;
		double VISUAL_avg_source_seg_len = 0;
		int VISUAL_source_longest_sent = 0;
		
		int VISUAL_num_target_word_tokens = 0;
		int VISUAL_total_target_vocab_size = 0;
		double VISUAL_avg_target_seg_len = 0;
		int VISUAL_target_longest_sent = 0;

		int VISUAL_num_segment_pairs = 0;
		int VISUAL_num_tokens_thrown_away = 0;

		//cout<<"test input_file_prep.h"<<endl;

		source_input.open(source_file_name.c_str()); 
		target_input.open(target_file_name.c_str());
		final_output.open(output_file_name.c_str()); //intermediate file: train.txt

		string src_str;
		string tgt_str; 
		string word;

		int source_len = 0;
		int target_len = 0;

		source_input.clear();
		target_input.clear();
	
		source_input.seekg(0, ios::beg);
		while(getline(source_input, src_str)) {
			source_len++;
		}

		target_input.seekg(0, std::ios::beg);
		while(getline(target_input, tgt_str)) {
			target_len++;
		}
		//sentence lengths
		VISUAL_num_segment_pairs = target_len;

		//filter any long sentences and get ready to shuffle
		source_input.clear();
		target_input.clear();
		source_input.seekg(0, ios::beg);
		target_input.seekg(0, ios::beg);
		for(int i=0; i<source_len; i++) {
			vector<std::string> src_sentence;
			vector<std::string> tgt_sentence;
			getline(source_input, src_str);
			getline(target_input, tgt_str);

			istringstream iss_src(src_str, istringstream::in);
			istringstream iss_tgt(tgt_str, istringstream::in);
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}
			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_avg_source_seg_len+=src_sentence.size();
				VISUAL_avg_target_seg_len+=tgt_sentence.size();
				VISUAL_num_source_word_tokens+=src_sentence.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}
		VISUAL_avg_source_seg_len = VISUAL_avg_source_seg_len/( (double)VISUAL_num_segment_pairs);
		VISUAL_avg_target_seg_len = VISUAL_avg_target_seg_len/( (double)VISUAL_num_segment_pairs);

		//cout<<"test 2 input_file_prep.h"<<endl;
		
		//shuffle the entire data
		if(MY_CUDA::shuffle_data) {
			random_shuffle(data.begin(),data.end());
		}

		//remove last sentences that do not fit in the minibatch
		if(data.size()%minibatch_size!=0) {
			int num_to_remove = data.size()%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}

		//sort the data based on minibatch
		compare_nonLM comp;
		int curr_index = 0;
		while(curr_index<data.size()) {
			if(curr_index+minibatch_size*minibatch_mult <= data.size()) {
				std::sort(data.begin()+curr_index,data.begin()+curr_index+minibatch_size*minibatch_mult,comp);
				curr_index+=minibatch_size*minibatch_mult;
			}
			else {
				std::sort(data.begin()+curr_index,data.end(),comp);
				break;
			}
		}

		//cout<<"test 3 input_file_prep.h"<<endl;
		
		//now get counts for mappings
		for(int i=0; i<data.size(); i++) {
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(data[i].src_sent[j]!= "<UNK>") {
					if(src_counts.count(data[i].src_sent[j])==0) {
						src_counts[data[i].src_sent[j]] = 1;
					}
					else {
						src_counts[data[i].src_sent[j]]+=1;
					}
				}
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(data[i].tgt_sent[j]!= "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {
						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}
		
		//cout<<"test 4 input_file_prep.h"<<endl;
		
		//now use heap to get the highest source and target mappings
		if(source_vocab_size == -1) {
			source_vocab_size = src_counts.size() + 1;
		}
		if(target_vocab_size == -1) {
			target_vocab_size = tgt_counts.size() + 3;
		}

		VISUAL_total_source_vocab_size = src_counts.size();
		VISUAL_total_target_vocab_size = tgt_counts.size();

		//vocab size
		source_vocab_size = min(source_vocab_size, (int)src_counts.size() + 1);
		target_vocab_size = min(target_vocab_size, (int)tgt_counts.size() + 1);

		//output the model info to first line of output weights file
		ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << num_layers << " " << embedding_size << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";

		priority_queue<mapping_pair,vector<mapping_pair>, mapping_pair_compare_functor> src_map_heap;
		priority_queue<mapping_pair,vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = src_counts.begin(); it != src_counts.end(); ++it ) {
			src_map_heap.push( mapping_pair(it->first,it->second) );
		}

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
		}
		//cout<<"test 4.1 input_file_prep.h"<<endl;
		
		//output the vocabulary
		output_model << "==========================================================\n";
		src_mapping["<UNK>"] = 0;
		output_model << 0 << " " << "<UNK>" << "\n";

		for(int i=1; i<source_vocab_size; i++) {
			src_mapping[src_map_heap.top().word] = i;
			output_model << i << " " << src_map_heap.top().word << "\n";
			src_map_heap.pop();
		}
		output_model << "==========================================================\n";

		//cout<<"test 4.2 input_file_prep.h"<<endl;
		
		tgt_mapping["<START>"] = 0;
		tgt_mapping["<EOF>"] = 1;
		tgt_mapping["<UNK>"] = 2;
		output_model << 0 << " " << "<START>" << "\n";
		output_model << 1 << " " << "<EOF>" << "\n";
		output_model << 2 << " " << "<UNK>" << "\n";

		for(int i=3; i<target_vocab_size; i++) {
			tgt_mapping[tgt_map_heap.top().word] = i;
			output_model << i << " " << tgt_map_heap.top().word << "\n";
			tgt_map_heap.pop();
		}
		output_model << "==========================================================\n";

		//cout<<"test 5 input_file_prep.h"<<endl;
		
		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
			std::vector<int> tgt_int;
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					src_int.push_back(src_mapping["<UNK>"]);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}
			std::reverse(src_int.begin(), src_int.end());
			data[i].src_sent.clear();
			data[i].src_sent_int = src_int;
			
			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {	
					tgt_int.push_back(tgt_mapping["<UNK>"]);
				}
				else {
					tgt_int.push_back(tgt_mapping[data[i].tgt_sent[j]]);
				}	
			}
			data[i].tgt_sent.clear();
			data[i].tgt_sent_int_i = tgt_int;
			data[i].tgt_sent_int_o = tgt_int;
			data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
			data[i].tgt_sent_int_o.push_back(1);
		}

		//now pad based on minibatch
		curr_index = 0;
		while(curr_index < data.size()) {
			int max_source_minibatch=0;
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].src_sent_int.size()>max_source_minibatch) {
					max_source_minibatch = data[i].src_sent_int.size();
				}
				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {

				while(data[i].src_sent_int.size()<max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
				}

				while(data[i].tgt_sent_int_i.size()<max_target_minibatch) {
					data[i].tgt_sent_int_i.push_back(-1);
					data[i].tgt_sent_int_o.push_back(-1);
				}
			}
			curr_index+=minibatch_size;
		}

		//cout<<"test 6 input_file_prep.h"<<endl;
		
		//now output to the file
		for(int i=0; i<data.size(); i++) {

			for(int j=0; j<data[i].src_sent_int.size(); j++) {
				final_output << data[i].src_sent_int[j];
				if(j!=data[i].src_sent_int.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].minus_two_source.size(); j++) {
				final_output << data[i].minus_two_source[j];
				if(j!=data[i].minus_two_source.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].tgt_sent_int_i.size(); j++) {
				final_output << data[i].tgt_sent_int_i[j];
				if(j!=data[i].tgt_sent_int_i.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";


			for(int j=0; j<data[i].tgt_sent_int_o.size(); j++) {
				final_output << data[i].tgt_sent_int_o[j];
				if(j!=data[i].tgt_sent_int_o.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";
		}

		final_output.close();
		source_input.close();
		target_input.close();

		//print file stats:
		cout << "----------------------------source train file info-----------------------------\n";
		cout << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		cout << "Source vocabulary size (before <unk>ing): " << VISUAL_total_source_vocab_size<<"\n";
		cout << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		cout << "Average source segment length: " << VISUAL_avg_source_seg_len<< "\n";
		cout << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		cout << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		cout << "----------------------------target train file info-----------------------------\n";
		cout << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		cout << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		cout << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		cout << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		cout << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		cout << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		cout << "-------------------------------------------------------------------------------\n\n";
	
		return true;
	}


	void integerize_file_nmt(string output_vocab_name, string source_file_name, string target_file_name, string tmp_output_name,
		int max_sent_cutoff, int minibatch_size, int &embedding_size, int &hiddenstate_size, int &source_vocab_size, int &target_vocab_size, int &num_layers)
	{

		int VISUAL_num_source_word_tokens =0;
		int VISUAL_source_longest_sent=0;

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_segment_pairs=0;
		int VISUAL_num_tokens_thrown_away=0;

		ifstream weights_file;
		weights_file.open(output_vocab_name.c_str());

		vector<string> file_input_vec;
		string str;
		string word;

		getline(weights_file, str);
		istringstream iss(str, istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		num_layers = stoi(file_input_vec[0]);
		embedding_size = stoi(file_input_vec[1]);
		hiddenstate_size = stoi(file_input_vec[2]);
		target_vocab_size = stoi(file_input_vec[3]);
		source_vocab_size = stoi(file_input_vec[4]);

		//now get the mappings
		std::getline(weights_file, str); //get this line, since all equals
		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			src_mapping[word] = tmp_index;
		}

		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			tgt_mapping[word] = tmp_index;
		}
		
		//now that we have the mappings, integerize the file
		std::ofstream final_output;
		final_output.open(tmp_output_name.c_str());
		std::ifstream source_input;
		source_input.open(source_file_name.c_str());
		std::ifstream target_input;
		target_input.open(target_file_name.c_str());
		
		//first get the number of lines the the files 
		int source_len = 0;
		int target_len = 0;
		std::string src_str;
		std::string tgt_str;

		source_input.clear();
		target_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}

		VISUAL_num_segment_pairs = target_len;
		
		source_input.clear();
		target_input.clear();
		source_input.seekg(0, ios::beg);
		target_input.seekg(0, ios::beg);
		for(int i=0; i<source_len; i++) {
			vector<string> src_sentence;
			vector<string> tgt_sentence;
			
			getline(source_input, src_str);
			getline(target_input, tgt_str);
			istringstream iss_src(src_str, istringstream::in);
			istringstream iss_tgt(tgt_str, istringstream::in);
			
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}

			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if(!(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2)) {
				data.push_back(comb_sent_info(src_sentence, tgt_sentence));
				VISUAL_num_source_word_tokens += src_sentence.size();
				VISUAL_num_target_word_tokens += tgt_sentence.size();
				
				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}

			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}	
		}

		if( (minibatch_size!=1) ) {
            if(MY_CUDA::shuffle_data) {
			    random_shuffle(data.begin(),data.end());
            }
		}

		//sort the data based on minibatch
		if( (minibatch_size!=1) ) {
			compare_nonLM comp;
			int curr_index = 0;
			while(curr_index<data.size()) {
				if(curr_index+minibatch_size*minibatch_mult <= data.size()) {
					std::sort(data.begin()+curr_index,data.begin()+curr_index+minibatch_size*minibatch_mult,comp);
					curr_index+=minibatch_size*minibatch_mult;
				}
				else {
					std::sort(data.begin()+curr_index,data.end(),comp);
					break;
				}
			}
		}

		if(data.size()%minibatch_size!=0) {
			int num_to_remove = data.size()%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}
		
		//now integerize
		for(int i=0; i<data.size(); i++) {
			vector<int> src_int;
			vector<int> tgt_int;

			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					src_int.push_back(src_mapping["<UNK>"]);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}

			reverse(src_int.begin(), src_int.end());
			data[i].src_sent.clear();
			data[i].src_sent_int = src_int;
			
			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
			}
			
			int max_iter = 0;
			max_iter = data[i].tgt_sent.size();
			for(int j=0; j<max_iter; j++) {
				
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					tgt_int.push_back(tgt_mapping["<UNK>"]);
				}
				else {
					tgt_int.push_back(tgt_mapping[data[i].tgt_sent[j]]);
				}
			}
			
			data[i].tgt_sent.clear();
			data[i].tgt_sent_int_i = tgt_int;
			data[i].tgt_sent_int_o = tgt_int;
			data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
			data[i].tgt_sent_int_o.push_back(1);
		}

		//now pad
		int curr_index = 0;
		while(curr_index < data.size()) {
			int max_source_minibatch=0;
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].src_sent_int.size()>max_source_minibatch) {
					max_source_minibatch = data[i].src_sent_int.size();
				}
				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {

				while(data[i].src_sent_int.size()<max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
				}

				while(data[i].tgt_sent_int_i.size()<max_target_minibatch) {
					data[i].tgt_sent_int_i.push_back(-1);
					data[i].tgt_sent_int_o.push_back(-1);
				}
			}
			curr_index+=minibatch_size;
		}

		//output to train(validation).txt
		for(int i=0; i<data.size(); i++) {

			for(int j=0; j<data[i].src_sent_int.size(); j++) {
				final_output << data[i].src_sent_int[j];
				if(j!=data[i].src_sent_int.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].minus_two_source.size(); j++) {
				final_output << data[i].minus_two_source[j];
				if(j!=data[i].minus_two_source.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].tgt_sent_int_i.size(); j++) {
				final_output << data[i].tgt_sent_int_i[j];
				if(j!=data[i].tgt_sent_int_i.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";


			for(int j=0; j<data[i].tgt_sent_int_o.size(); j++) {
				final_output << data[i].tgt_sent_int_o[j];
				if(j!=data[i].tgt_sent_int_o.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";
		}

		weights_file.close();
		final_output.close();
		source_input.close();
		target_input.close();


		//print file stats:
		cout << "----------------------------source dev file info-----------------------------\n";
		cout << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		cout << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		cout << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		cout << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		cout << "----------------------------target dev file info-----------------------------\n\n";
		cout << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		cout << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		cout << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		cout << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		cout << "-------------------------------------------------------------------------------\n\n";
	}

};

#endif
