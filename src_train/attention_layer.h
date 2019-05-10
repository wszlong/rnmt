
#ifndef ATTENTION_LAYER_H
#define ATTENTION_LAYER_H

template<typename dType>
class neuralMT_model;

template<typename dType>
class attention_node;

#include "gpu_info_struct.h"

template<typename dType>
class attention_layer {
public:

	cublasHandle_t handle;
	int device_number;
	int Embedding_size;
	int LSTM_size;
	int minibatch_size;
	bool clip_gradients;
	dType norm_clip;
	bool feed_input = false;
	int longest_sent;
	bool transfer_done = false; //if true then take the copied matrix
	
	//adam
	bool ADAM;
	dType alpha_adam;
	dType beta_1;
	dType beta_2;
	dType epsilon;
	//adam
	dType *d_W_c_p1_mt;
	dType *d_W_c_p1_vt;
	dType *d_W_c_p2_mt;
	dType *d_W_c_p2_vt;
	dType *d_output_bias_mt;
	dType *d_output_bias_vt;
	
	dType *d_W_a; //for the score function
	dType *d_W_p;
	dType *d_W_c_p1;
	dType *d_W_c_p2;
	dType *d_output_bias;
	
	dType *d_W_c_p1_grad;
	dType *d_W_c_p2_grad;
	dType *d_output_bias_grad;
	
	thrust::device_ptr<dType> thrust_d_W_c_p1_grad;
	thrust::device_ptr<dType> thrust_d_W_c_p2_grad;
	thrust::device_ptr<dType> thrust_d_output_bias_grad;
	
	dType *d_ERRnTOt_ht_p1;
	dType *d_ERRnTOt_tan_htild;
	dType *d_ERRnTOt_ct;

	dType *d_ERRnTOt_as;
	dType *d_ERRnTOt_hsht;
	dType *d_ERRnTOt_pt;
	dType *d_ERRnTOt_htild_below; //from the input layer
	
	dType *d_result; //for gradient clipping
	dType *d_temp_result; // for gradient clipping
		
	dType **d_total_hs_mat;
	dType **d_total_hs_error;
	dType *d_ones_minibatch;
	dType *d_temp_1; //LSTM by minibatch size
	dType *d_h_t_sum; //for summing weighted h_t's
	dType *d_h_s_sum; //for summing weighted h_s
	
	attention_layer_gpu_info layer_info; //stores the gpu info for the attention model
	curandGenerator_t rand_gen;

	int *d_batch_info; // length of minibatches, then offsets

	int *d_ones_minibatch_int;

	std::vector<attention_node<dType>> nodes;
	neuralMT_model<dType> *model;

	attention_layer() {};

	attention_layer(int Embedding_size,int LSTM_size,int minibatch_size, int device_number, int longest_sent,cublasHandle_t &handle,neuralMT_model<dType> *model,
		bool feed_input,bool clip_gradients,dType norm_clip,bool dropout,dType dropout_rate,global_params &params);
	
	void prep_minibatch_info(int *h_batch_info);

	void clear_gradients();
	void scale_gradients();
	void clip_gradients_func();
	void update_params(int time_index);

	void dump_weights(ofstream &output);

	void load_weights(ifstream &input);

	void check_gradients(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
};

#endif
