
#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "LSTM.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include "softmax.h"
#include <math.h>
#include <limits>
#include "Input_To_Hidden_Layer.h"
#include "Hidden_To_Hidden_Layer.h"

using namespace std;

template<typename dType>
class Input_To_Hidden_Layer;

template<typename dType>
class Hidden_To_Hidden_Layer;

struct file_helper;

template<typename dType>
class neuralMT_model {
	public:

	base_loss_layer<dType> *softmax;
	
	//First layer of model, the input to hidden layer
	Input_To_Hidden_Layer<dType> input_layer_source;
	Input_To_Hidden_Layer<dType> input_layer_source_bi;
	Input_To_Hidden_Layer<dType> input_layer_target;

	//Hidden layers of model
	vector<Hidden_To_Hidden_Layer<dType>> source_hidden_layers;
	vector<Hidden_To_Hidden_Layer<dType>> target_hidden_layers;
	
	file_helper *file_info;
	softmax_layer_gpu_info s_layer_info;
	
	ifstream input;
	ofstream output;
	
	string input_weight_file;
	string output_vocab_file; 
	string output_weight_file; 
	
	bool debug;

	bool train_perplexity_mode;
	double train_perplexity=0;
	
	bool train = false;	
	bool grad_check_flag = false;	

	//for the attention model
	int source_length = -1;
	
	//attention model
	attention_params attent_params;
	ofstream output_alignments;
	
	neuralMT_model() {};

	//Called at beginning of program once to initialize the weights
	void initModel(int Embedding_size,int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
		int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
		string input_weight_file,string output_vocab_file,string output_weight_file,bool train_perplexity,
		int num_layers,vector<int> gpu_indicies,bool dropout,dType dropout_rate,
		struct attention_params attent_params,global_params &params);
	
	//Maps the file info pointer to the model
	void initFileInfo(struct file_helper *file_info);
	
	//Dumps all the GPU info
	void print_GPU_Info();
	
	void compute_gradients(int *h_input_vocab_indicies_source,int *h_input_vocab_indicies_source_bi,
		int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,
		int *h_output_vocab_indicies_target,int current_source_length,int current_target_length,
		int *h_output_vocab_indicies_source_Wgrad,int *h_input_vocab_indicies_target_Wgrad,
		int len_source_Wgrad,int len_target_Wgrad,int *h_batch_info,file_helper *temp_fh,int time_index);
	
	//Sets all gradient matrices to zero, called after a minibatch updates the gradients
	void clear_gradients();
	
	//Called after each minibatch, once the gradients are calculated
	void update_weights(int time_index);


	//gets the perplexity of a file
	double get_perplexity(string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
		int source_vocab_size,int target_vocab_size,int &test_total_words);
	//Get the sum of all errors in the minibatch
	double getError();

	void update_learning_rate(dType new_learning_rate);
	void adam_switch_sgd();
	
	//Output the weights to a file
	void dump_weights();

	void dump_best_model(string best_model_name,string const_model);
	
	//Read in Weights from file
	void load_weights();
	
	void check_all_gradients(dType epsilon);
};

#endif
