
#ifndef LSTM_HIDDEN_TO_HIDDEN_H
#define LSTM_HIDDEN_TO_HIDDEN_H

#include "LSTM_HH.h"
#include "transfer_layer.h"

template<typename dType>
class neuralMT_model;

//template<typename dType>
//class attention_layer;

template<typename dType>
class Hidden_To_Hidden_Layer {
public:
	
	vector<LSTM_HH_Node<dType>> nodes;
	layer_gpu_info hh_layer_info;
	
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
	int current_length; //This is the current length of this target or source sequence

	int *h_input_vocab_indices_01_full; //only for debugging
	int *d_input_vocab_indices_01_full;

	//for setting inital cell and hidden state values
	dType *h_init_hidden_vector;
	dType *h_init_cell_vector;
	dType *d_init_hidden_vector;
	dType *d_init_cell_vector;

	dType *h_init_d_ERRnTOtp1_ht;
	dType *h_init_d_ERRnTOtp1_ct;
	dType *d_init_d_ERRnTOtp1_ht;
	dType *d_init_d_ERRnTOtp1_ct;
	
	//stuff for norm clipping
	dType *d_result;
	dType *d_temp_result;


	//device pointers
	dType *d_temp1;
	dType *d_temp2;
	dType *d_temp3;
	dType *d_temp4;
	dType *d_temp5;
	dType *d_temp6;
	dType *d_temp7;
	dType *d_temp8;
	//bi
	dType *d_temp1_bi;
	dType *d_temp3_bi;
	dType *d_temp5_bi;
	dType *d_temp7_bi;

	dType *d_W_ho;
	dType *d_W_hf;
	dType *d_W_hi;
	dType *d_W_hc;
	dType *d_W_hi_grad;
	dType *d_W_hf_grad;
	dType *d_W_hc_grad;
	dType *d_W_ho_grad;

	dType *d_ones_minibatch;

	dType *d_M_i;
	dType *d_M_f;
	dType *d_M_o;
	dType *d_M_c;
	dType *d_M_i_grad;
	dType *d_M_f_grad;
	dType *d_M_o_grad;
	dType *d_M_c_grad;
	
	//bi
	dType *d_U_i;
	dType *d_U_f;
	dType *d_U_o;
	dType *d_U_c;
	dType *d_U_i_grad;
	dType *d_U_f_grad;
	dType *d_U_o_grad;
	dType *d_U_c_grad;

	dType *d_b_i;
	dType *d_b_f;
	dType *d_b_c;
	dType *d_b_o;
	dType *d_b_i_grad;
	dType *d_b_f_grad;
	dType *d_b_c_grad;
	dType *d_b_o_grad;

	dType *h_h_t_below;
	dType *d_h_t_below;
	
	//thrust device pointers to doing parameter updates nicely (not input word embeddings though)
	thrust::device_ptr<dType> thrust_d_W_ho_grad; 
	thrust::device_ptr<dType> thrust_d_W_hf_grad;
	thrust::device_ptr<dType> thrust_d_W_hi_grad; 
	thrust::device_ptr<dType> thrust_d_W_hc_grad;

	thrust::device_ptr<dType> thrust_d_M_i_grad;
	thrust::device_ptr<dType> thrust_d_M_f_grad;
	thrust::device_ptr<dType> thrust_d_M_o_grad;
	thrust::device_ptr<dType> thrust_d_M_c_grad;
	//bi
	thrust::device_ptr<dType> thrust_d_U_i_grad;
	thrust::device_ptr<dType> thrust_d_U_f_grad;
	thrust::device_ptr<dType> thrust_d_U_o_grad;
	thrust::device_ptr<dType> thrust_d_U_c_grad;

	thrust::device_ptr<dType> thrust_d_b_i_grad;
	thrust::device_ptr<dType> thrust_d_b_f_grad;
	thrust::device_ptr<dType> thrust_d_b_c_grad;
	thrust::device_ptr<dType> thrust_d_b_o_grad;

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
	dType *h_d_ERRnTOt_h_Below;

	dType *d_d_ERRnTOt_ht;
	dType *d_d_ERRt_ct;
	dType *d_d_ERRnTOt_ct;
	dType *d_d_ERRnTOt_ot;
	dType *d_d_ERRnTOt_ft;
	dType *d_d_ERRnTOt_tanhcpt;
	dType *d_d_ERRnTOt_it;
	dType *d_d_ERRnTOt_htM1;
	dType *d_d_ERRnTOt_ctM1;
	dType *d_d_ERRnTOt_h_Below;
	dType *d_d_ERRnTOt_h_Below_bi; //

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
	//bidire
	dType *d_U_i_mt;
	dType *d_U_i_vt;
	dType *d_U_f_mt;
	dType *d_U_f_vt;
	dType *d_U_c_mt;
	dType *d_U_c_vt;
	dType *d_U_o_mt;
	dType *d_U_o_vt;
	
	dType *d_b_i_mt;
	dType *d_b_i_vt;
	dType *d_b_f_mt;
	dType *d_b_f_vt;
	dType *d_b_c_mt;
	dType *d_b_c_vt;
	dType *d_b_o_mt;
	dType *d_b_o_vt;
	
	///////////////////////////////////Other parameters///////////////////////////////////
	boost::random::mt19937 gen; //Random number generator for initializing weights

	neuralMT_model<precision> *model;
	

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
	int layer_number = -1; //start at 1, for indexing directly into the target layer
	curandGenerator_t rand_gen;
	attention_layer<dType> *attent_layer=NULL;
	
	upper_transfer_layer<dType> upper_layer;
	lower_transfer_layer<dType> lower_layer;

	Hidden_To_Hidden_Layer() {};

	void init_Hidden_To_Hidden_Layer(int Embedding_size,int LSTM_size,int minibatch_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,int seed,bool dropout,dType dropout_rate,int layer_number,global_params &params);

	void init_Hidden_To_Hidden_Layer_GPU(int Embedding_size,int LSTM_size,int minibatch_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,int seed);
	
	void init_attention(int device_number,bool feed_input,neuralMT_model<dType> *model,global_params &params);
	
	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_input_vocab_indicies,int current_length);

	//Clear the previous gradients
	void clear_gradients(bool init);
	void clear_gradients_GPU(bool init);

	//Update the weights of the model
	void update_weights(int time_index);
	void update_weights_GPU(int time_index);
	
	void zero_attent_error();
	void scale_gradients();
	void update_params(int time_index);

	void dump_weights(ofstream &output);
	void dump_weights_GPU(ofstream &output);

	void load_weights(ifstream &input);
	void load_weights_GPU(ifstream &input);

	void check_all_gradients(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

};

#endif
