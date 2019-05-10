
#ifndef LSTM_IH_H
#define LSTM_IH_H

#include "model.h"

template<typename dType>
class neuralMT_model;

template<typename dType>
class Input_To_Hidden_Layer;


template<typename dType>
class LSTM_IH_Node {
public:
	//Pointer to the model struct, so it can access all of the weight matrices
	Input_To_Hidden_Layer<precision> *model;
	
	//--------------------------------------------------GPU parameters------------------------------------
	int minibatch_size;
	int Embedding_size;
	int LSTM_size;
	int index;
	bool feed_input = false;
	

	bool bi_dir;
	int current_length;
	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;


	//host pointers
	dType *h_o_t;
	dType *h_c_t;
	int *h_input_vocab_indices_01;
	int *h_input_vocab_indices;
	dType *h_f_t;
	dType *h_c_t_prev;
	dType *h_c_prime_t_tanh;
	dType *h_i_t;
	dType *h_h_t_prev;

	dType *h_sparse_lookup;
	dType *h_h_t;
	
	dType *h_d_ERRt_ht;

	//device pointers
	dType *d_o_t;
	dType *d_c_t;
	int *d_input_vocab_indices_01;
	int *d_input_vocab_indices;
	dType *d_f_t;
	dType *d_c_t_prev;
	dType *d_c_prime_t_tanh;
	dType *d_i_t;

	dType *d_h_t_prev;
	dType *d_sparse_lookup;
	dType *d_h_t;
	dType *d_zeros; //points to a zero matrix that can be used for d_ERRt_ht in backprop
	dType *d_h_tild;

	dType *d_d_ERRnTOtp1_ht;
	dType *d_d_ERRnTOtp1_ct;
	dType *d_d_ERRt_ht;

	dType *d_ERRnTOt_h_tild;
	dType *d_ERRnTOt_h_tild_cpy;
	
	//Constructor
	LSTM_IH_Node(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m,int index,bool dropout,dType dropout_rate,bool bi_dir);

	void init_LSTM_GPU(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m);

	void attention_extra();
	
	void update_vectors_forward_GPU(int *d_input_vocab_indices,int *d_input_vocab_indices_01,
		dType *d_h_t_prev,dType *d_c_t_prev, int current_length);

	//Compute the forward values for the LSTM node
	void forward_prop();
	void forward_prop_GPU();

	void send_h_t_above();

	void backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct);//,dType *d_d_ERRt_ht);
	void back_prop_GPU(int index);

	//Update the gradient matrices
	void compute_gradients_GPU();

};

#endif
