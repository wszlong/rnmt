
#ifndef ATTENTION_NODE_H
#define ATTENTION_NODE_H


template<typename dType>
class attention_layer;


template<typename dType>
class attention_node {
public:

	attention_layer<dType> *attent_layer;
	int Embedding_size;
	int LSTM_size;
	int minibatch_size;
	int device_number;
	
	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;

	dType *d_tanh_1;
	dType *d_alignments; // size (minibatch size x 2D + 1)
	dType *d_normal_alignments;
	dType *d_h_t=NULL;
	dType *d_c_t; // size (LSTM size, minibatch size)
	dType *d_final_temp_1; //for W_c_p1*c_t
	dType *d_final_temp_2; //for W_c_p2*h_t, also reuse to add the bias and tanh
//	int *d_lower_upper;
	int *d_indicies;
	dType *d_h_t_att;

	int **d_indicies_mask; //points to the LSTM node for this info for zeroing out forward and back prop
	dType *d_hs_mat;
	
	dType *d_lower_htild; //send htild to this location
	dType *d_ERR_above; //what the LSTM get passed from the above layer or softmax
	
	dType *d_d_ERRt_ht_tild; //this is the error passed back from the softmax
	dType *d_d_ERRt_ht_input; //if feed input, then this error will be added in place to d_d_ERRt_ht_p
	dType *d_ERRtTOn_htild_below; //this is from the previous lower LSTM for feed input
	
	bool feed_input = false; //get rid of most parallelism :( 
	int index;

	attention_node(int Embedding_size,int LSTM_size,int minibatch_size,int device_number,bool feed_input,attention_layer<dType> *attent_layer,int index,bool dropout,dType dropout_rate);
	
	void feed_input_init(dType *d_ptr_htild);
	
	void forward_prop();
	
	void back_prop();

};

#endif
