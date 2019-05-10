
#ifndef LSTM_HH_H
#define LSTM_HH_H

#include "Util.h"
#include "model.h"

//Forward declaration
template<typename dType>
class neuralMT_model;

template<typename dType>
class Hidden_To_Hidden_Layer;

template<typename dType>
class LSTM_HH_Node {
public:
	
	//--------------------------------------------------GPU parameters------------------------------------
	int minibatch_size;
	int Embedding_size;
	int LSTM_size;
	int index; //what node is this
	bool attention_model = false; //this will only be true for the upper layer on the target side of the LSTM

	int current_length;	
	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;

	//Pointer to the model struct, so it can access all of the weight matrices
	Hidden_To_Hidden_Layer<precision> *model;
	
	//host pointers
	dType *h_o_t;
	dType *h_c_t;
	int *h_input_vocab_indices_01;
	dType *h_f_t;
	dType *h_c_t_prev;
	dType *h_c_prime_t_tanh;
	dType *h_i_t;
	dType *h_h_t_prev;
	dType *h_h_t;
	
	dType *h_d_ERRt_ht;

	//device pointers
	dType *d_o_t;
	dType *d_c_t;
	int *d_input_vocab_indices_01;
	dType *d_f_t;
	dType *d_c_t_prev;
	dType *d_c_prime_t_tanh;
	dType *d_i_t;
	dType *d_h_t_prev;
	dType *d_h_t;
	
	dType *h_h_t_below;
	dType *d_h_t_below;
	dType *d_h_t_below_bi; //

	//dType *d_zeros;
	dType *d_h_tild;
	
	dType *d_ERRnTOt_h_tild;
	dType *d_d_ERRnTOtp1_ht;
	dType *d_d_ERRnTOtp1_ct;
	dType *d_d_ERRt_ht;
	
	//Constructor
	LSTM_HH_Node(int Embedding_size,int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m,int index,bool dropout,dType dropout_rate);

	void init_LSTM_GPU(int Embedding_size,int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m);

	void update_vectors_forward_GPU(int *d_input_vocab_indices_01,dType *d_h_t_prev,dType *d_c_t_prev, int current_length);

	//Compute the forward values for the LSTM node
	void forward_prop();
	void forward_prop_GPU();

	void forward_prop_sync(cudaStream_t &my_s);
	void forward_prop_sync_bi(cudaStream_t &my_s);
	void send_h_t_above();

	void backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct);
	void back_prop_GPU(int index);

	//Update the gradient matrices
	void compute_gradients_GPU();


};


#endif
