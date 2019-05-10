
#ifndef LSTM_INPUT_TO_HIDDEN_H
#define LSTM_INPUT_TO_HIDDEN_H

#include "transfer_layer.h"

template<typename dType>
class neuralMT_model;

template<typename dType>
class attention_layer;

template<typename dType>
class Input_To_Hidden_Layer {
public:
	
	vector<LSTM_IH_Node<dType>> nodes;
	layer_gpu_info ih_layer_info;
	
	/////////////////////////////GPU parameters/////////////////////////////////////

	//host pointers
	dType *h_temp1;
	dType *h_temp2;
	dType *h_temp3;
	dType *h_temp4;
	dType *h_temp5;
	dType *h_temp6;
	dType *h_temp7;
	dType *h_temp8;

	dType *h_W_ho;
	dType *h_W_hf;
	dType *h_W_hi;
	dType *h_W_hc;
	dType *h_W_hi_grad;
	dType *h_W_hf_grad;
	dType *h_W_hc_grad;
	dType *h_W_ho_grad;
	
	dType *h_W;
	dType *h_W_grad;
	dType *h_ones_minibatch;
	
	dType *h_M_i;
	dType *h_M_f;
	dType *h_M_o;
	dType *h_M_c;
	dType *h_M_i_grad;
	dType *h_M_f_grad;
	dType *h_M_o_grad;
	dType *h_M_c_grad;

	dType *h_b_i;
	dType *h_b_f;
	dType *h_b_c;
	dType *h_b_o;
	dType *h_b_i_grad;
	dType *h_b_f_grad;
	dType *h_b_c_grad;
	dType *h_b_o_grad;


	//Convert this into 0/1's and to one with no -1's as indicies
	int *h_input_vocab_indicies;
	int *d_input_vocab_indicies; 
	int *h_input_vocab_indicies_bi; //
	int *d_input_vocab_indicies_bi; //

	int current_length; //This is the current length of this target or source sequence
	int w_grad_len; //This is special length for the W_grad special preprocessing for vocab indicies
	
	int *h_input_vocab_indices_full; //only for debugging
	int *h_input_vocab_indices_01_full; //only for debugging
	int *d_input_vocab_indices_full;
	int *d_input_vocab_indices_01_full;
	
	int *h_input_vocab_indicies_Wgrad; //for update word embedding
	int *d_input_vocab_indicies_Wgrad;
	
	
	//for setting inital cell and hidden state values
	dType *h_init_hidden_vector;
	dType *h_init_cell_vector;
	dType *d_init_hidden_vector;
	dType *d_init_cell_vector;

	//stuff for norm clipping
	dType *d_result;
	dType *d_temp_result;

	dType *h_init_d_ERRnTOtp1_ht;
	dType *h_init_d_ERRnTOtp1_ct;
	dType *d_init_d_ERRnTOtp1_ht;
	dType *d_init_d_ERRnTOtp1_ct;
	
	
	//device pointers
	dType *d_temp1;
	dType *d_temp2;
	dType *d_temp3;
	dType *d_temp4;
	
	dType *d_temp5;
	dType *d_temp6;
	dType *d_temp7;
	dType *d_temp8;

	dType *d_temp9;
	dType *d_temp10;
	dType *d_temp11;
	dType *d_temp12;

	dType *d_W_ho;
	dType *d_W_hf;
	dType *d_W_hi;
	dType *d_W_hc;
	dType *d_W_hi_grad;
	dType *d_W_hf_grad;
	dType *d_W_hc_grad;
	dType *d_W_ho_grad;

	dType *d_W;
	dType *d_ones_minibatch;

	dType *d_M_i;
	dType *d_M_f;
	dType *d_M_o;
	dType *d_M_c;
	dType *d_M_i_grad;
	dType *d_M_f_grad;
	dType *d_M_o_grad;
	dType *d_M_c_grad;

	dType *d_b_i;
	dType *d_b_f;
	dType *d_b_c;
	dType *d_b_o;
	dType *d_b_i_grad;
	dType *d_b_f_grad;
	dType *d_b_c_grad;
	dType *d_b_o_grad;

	dType *d_small_W_grad;
	int *d_reverse_unique_indicies;
	
	//these are for the feed input connections
	dType *d_Q_i;
	dType *d_Q_f;
	dType *d_Q_o;
	dType *d_Q_c;
	dType *d_Q_i_grad;
	dType *d_Q_f_grad;
	dType *d_Q_o_grad;
	dType *d_Q_c_grad;

	//new for saving space in the LSTM
	dType *h_d_ERRnTOt_ht;
	dType *h_d_ERRt_ct;
	dType *h_d_ERRnTOt_ct;
	dType *h_d_ERRnTOt_ot;
	dType *h_d_ERRnTOt_ft;
	dType *h_d_ERRnTOt_tanhcpt;
	dType *h_d_ERRnTOt_it;
	dType *h_d_ERRnTOt_htM1;
	dType *h_d_ERRnTOt_ctM1;

	dType *d_d_ERRnTOt_ht;
	dType *d_d_ERRt_ct;
	dType *d_d_ERRnTOt_ct;
	dType *d_d_ERRnTOt_ot;
	dType *d_d_ERRnTOt_ft;
	dType *d_d_ERRnTOt_tanhcpt;
	dType *d_d_ERRnTOt_it;
	dType *d_d_ERRnTOt_htM1;
	dType *d_d_ERRnTOt_ctM1;
	
	//using: scale-gridients; check_gridients; get_perplexity; clip_gradients
	//thrust device pointers to doing parameter updates nicely (not input word embeddings though)
	thrust::device_ptr<dType> thrust_d_W_ho_grad; 
	thrust::device_ptr<dType> thrust_d_W_hf_grad;
	thrust::device_ptr<dType> thrust_d_W_hi_grad; 
	thrust::device_ptr<dType> thrust_d_W_hc_grad;

	thrust::device_ptr<dType> thrust_d_M_i_grad;
	thrust::device_ptr<dType> thrust_d_M_f_grad;
	thrust::device_ptr<dType> thrust_d_M_o_grad;
	thrust::device_ptr<dType> thrust_d_M_c_grad;

	thrust::device_ptr<dType> thrust_d_Q_i_grad;
	thrust::device_ptr<dType> thrust_d_Q_f_grad;
	thrust::device_ptr<dType> thrust_d_Q_o_grad;
	thrust::device_ptr<dType> thrust_d_Q_c_grad;

	//remove then put in custom reduction kernel
	thrust::device_ptr<dType> thrust_d_W_grad;
	thrust::device_ptr<dType> thrust_d_small_W_grad;

	thrust::device_ptr<dType> thrust_d_b_i_grad;
	thrust::device_ptr<dType> thrust_d_b_f_grad;
	thrust::device_ptr<dType> thrust_d_b_c_grad;
	thrust::device_ptr<dType> thrust_d_b_o_grad;

	
	///////////////////////////OTher parameters////////////////////////////////////

	boost::random::mt19937 gen; //Random number generator for initializing weights

	neuralMT_model<precision> *model;
	
	bool bi_dir = false;
	//for dropout
	bool dropout;
	dType dropout_rate;

	bool debug; //True if want debugging printout,false otherwise
	int minibatch_size;
	dType learning_rate;
	bool clip_gradients;
	dType norm_clip; //For gradient clipping
	int Embedding_size;
	int LSTM_size;
	int longest_sent;
	int input_vocab_size;

	//adam
	bool ADAM = false;
	dType alpha_adam;
	dType beta_1;
	dType beta_2;
	dType epsilon;
	
	//adam	
	dType *d_W_hi_mt;
	dType *d_W_hi_vt;
	dType *d_W_hf_mt;
	dType *d_W_hf_vt;
	dType *d_W_hc_mt;
	dType *d_W_hc_vt;
	dType *d_W_ho_mt;
	dType *d_W_ho_vt;
	
	dType *d_M_i_mt;
	dType *d_M_i_vt;
	dType *d_M_f_mt;
	dType *d_M_f_vt;
	dType *d_M_c_mt;
	dType *d_M_c_vt;
	dType *d_M_o_mt;
	dType *d_M_o_vt;
	
	dType *d_b_i_mt;
	dType *d_b_i_vt;
	dType *d_b_f_mt;
	dType *d_b_f_vt;
	dType *d_b_c_mt;
	dType *d_b_c_vt;
	dType *d_b_o_mt;
	dType *d_b_o_vt;

	dType *d_W_mt;
	dType *d_W_vt;
	
	dType *d_Q_i_mt;
	dType *d_Q_i_vt;
	dType *d_Q_f_mt;
	dType *d_Q_f_vt;
	dType *d_Q_o_mt;
	dType *d_Q_o_vt;
	dType *d_Q_c_mt;
	dType *d_Q_c_vt;
	

	attention_layer<dType> *attent_layer=NULL;
	bool feed_input = false;
	curandGenerator_t rand_gen;
	
	upper_transfer_layer<dType> upper_layer;

	Input_To_Hidden_Layer() {};
	
	void init_Input_To_Hidden_Layer(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,
		int seed,bool dropout,dType dropout_rate,bool bi_dir,global_params &params,bool source);
	
	void init_Input_To_Hidden_Layer_GPU(int Embedding_size, int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,
		int seed,global_params &params,bool source);

	void init_feed_input(Hidden_To_Hidden_Layer<dType> *hidden_layer);

	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length,int len_W);
	void prep_GPU_vocab_indices_bi(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length,int len_W);


	//Clear the previous gradients
	void clear_gradients(bool init);
	void clear_gradients_GPU(bool init);

	//Update the weights of the model
	void update_weights(int time_index);
	void update_weights_GPU(int time_index);

	void scale_gradients();
	void update_params(int time_index);

	void dump_weights(ofstream &output);
	void dump_weights_GPU(ofstream &output);

	void load_weights(ifstream &input);
	void load_weights_GPU(ifstream &input);

	void check_all_gradients(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	void check_gradient_GPU_SPARSE(dType epsilon,dType *d_mat,dType *d_grad,int LSTM_size,int *h_unique_indicies,int curr_num_unique);
	void check_gradient_GPU_SPARSE_bi(dType epsilon,dType *d_mat,dType *d_grad,dType *d_grad_2,int LSTM_size,int *h_unique_indicies,int curr_num_unique);

};

#endif
