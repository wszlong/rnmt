
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "gpu_info_struct.h"
#include "softmax_node.h"
#include "transfer_layer.h"
#include "base_loss.h"

template<typename dType>
class softmax_layer : public base_loss_layer<dType> {
public:
	softmax_layer_gpu_info s_layer_info;
	
	//host pointers
	dType *h_D;
	dType *h_h_t;
	dType *h_b_d;
	dType *h_ones;
	int *h_output_vocab_indices;
	int *h_output_vocab_indices_01;
	dType *h_output_vocab_indices_01_float;
	
	dType *h_d_ERRt_ht;
	dType *h_D_grad;
	dType *h_b_d_grad;

	//device pointers
	dType *d_D;
	dType *d_h_t;
	dType *d_b_d;
	dType *d_ones;
	int *d_output_vocab_indices;
	int *d_output_vocab_indices_01;
	dType *d_output_vocab_indices_01_float;
	dType *d_outputdist;
	dType *d_normalization;

	dType *d_d_ERRt_ht;
	dType *d_D_grad;
	dType *d_b_d_grad;
	
	thrust::device_ptr<dType> thrust_d_D_grad; 
	thrust::device_ptr<dType> thrust_d_b_d_grad;
	
	thrust::host_vector<dType> thrust_h_outputdist;
	thrust::host_vector<dType> thrust_h_normalization;

	thrust::device_vector<dType> thrust_d_outputdist;
	thrust::device_vector<dType> thrust_d_normalization;

	double *d_train_perplexity;
	double *d_outputdist_perp;
	//for norm clipping
	dType *d_result;
	dType *d_temp_result;

	//These are simply pointers to the non-single versions, since the full versions contain the indicies for the whole minibatch
	int *d_output_vocab_indices_single;
	int *d_output_vocab_indices_01_single;
	dType *d_output_vocab_indices_01_float_single;

	boost::random::mt19937 gen; //Random number generator for initializing weights

	bool clip_gradients; //If true then clip gradients
	dType norm_clip; //For gradient clipping
	int minibatch_size;
	int output_vocab_size;
	int Embedding_size;
	int LSTM_size;
	dType learning_rate;
	bool train_perplexity;

	bool dropout;
	dType dropout_rate;
	
	//adam
	bool ADAM = false;
	dType alpha_adam;
	dType beta_1;
	dType beta_2;
	dType epsilon;
	
	//adam
	dType *d_D_mt;
	dType *d_D_vt;
	dType *d_b_d_mt;
	dType *d_b_d_vt;

	neuralMT_model<precision> *model;
	lower_transfer_layer<dType> lower_layer;
	vector<softmax_node<dType>> nodes;
	
	curandGenerator_t rand_gen;
	
	softmax_layer() {};
	
	softmax_layer_gpu_info gpu_init(int device_number);

	void init_loss_layer(struct neuralMT_model<precision> *model,global_params &params); 

	void init_softmax_layer_GPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int Embedding_size,int LSTM_size, bool clip_gradients,dType learning_rate,int longest_sent);

	void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer);
	
	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length);

	void backprop_prep_GPU(dType *d_h_t,int step);
	void backprop_prep_GPU_mgpu(int step);
	
	void forward_prop(int index);
	void forward_prop_GPU(int index);
	
	dType *get_ht_ptr(int index);
	void set_ht_ptr(int index,dType *d_h_t);
	
	void get_distribution_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D,dType *d_b_d,dType *d_h_t);

	void back_prop1(int index);
	void back_prop1_GPU(int index);

	void back_prop2(int index);
	void back_prop2_GPU(int index);

	void get_h_t_gradient_GPU(int output_vocab_size,dType *d_D,dType *d_outputdist,dType *d_d_ERRt_ht,int index);

	void compute_D_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D_grad,dType *d_h_t);

	void compute_b_d_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_b_d_grad);

	cudaEvent_t get_ERR_ht_event();

	double get_train_perplexity();

	void clear_gradients();
	void clear_gradients_GPU();

	void update_weights(int time_index);
	void update_weights_GPU(int time_index);

	void update_learning_rate(dType learning_rate);
	void adam_switch_sgd();

	void dump_weights(ofstream &output);
	void dump_weights_GPU(ofstream &output);

	void load_weights(ifstream &input);
	void load_weights_GPU(ifstream &input);

	double compute_loss_GPU(int index);
	
	void get_perplexity_GPU(dType *d_h_t,int index); 
	
	void check_all_gradients(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

};

#endif
