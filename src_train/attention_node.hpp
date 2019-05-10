
template<typename dType>
attention_node<dType>::attention_node(int Embedding_size,int LSTM_size,int minibatch_size,int device_number,bool feed_input,attention_layer<dType> *attent_layer,int index,bool dropout,dType dropout_rate) {

	this->Embedding_size = Embedding_size;	
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->device_number = device_number;
	this->feed_input = feed_input;
	this->attent_layer = attent_layer;
	this->index = index;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;
 
	cudaSetDevice(device_number);
	
	//CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_tanh_1, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_c_t, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_temp_1, Embedding_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_temp_2, Embedding_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_att, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");

	if(dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	}
	
	dType *h_temp;
	full_matrix_setup_0(&h_temp,&d_alignments,minibatch_size,attent_layer->longest_sent);
	full_matrix_setup_0(&h_temp,&d_normal_alignments,minibatch_size,attent_layer->longest_sent);
	full_matrix_setup_0(&h_temp,&d_c_t,LSTM_size,minibatch_size);

	this->index = index;
}


template<typename dType>
void attention_node<dType>::feed_input_init(dType *d_ptr_htild) {
	cudaSetDevice(device_number);
	d_lower_htild = d_ptr_htild; // nodes[i+1].d_h_tild
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_ERRtTOn_htild_below, Embedding_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
}

template<typename dType>
void attention_node<dType>::forward_prop() {
	
	cudaSetDevice(device_number);
	cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.start_forward,0);

	cudaMemcpyAsync(d_h_t_att, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,attent_layer->layer_info.s0); //d_h_t --> target_hidden_layer[].node.d_h_t
	
	if(dropout && attent_layer->model->train) {

		if(!attent_layer->model->grad_check_flag) {
			curandSetStream(attent_layer->rand_gen,attent_layer->layer_info.s0);
			curandGenerateUniform_wrapper(d_dropout_mask,LSTM_size*minibatch_size,attent_layer->rand_gen); 
		}
		dropout_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(d_dropout_mask,dropout_rate,d_h_t_att,LSTM_size*minibatch_size);
	}

	
	dType alpha = 1;
	dType beta = 0;

	//compute attention weights
	//h_total_hs_mat --> source_hidden_layers.size()-1].nodes[i].d_h_t;
	alignment_kernel<<<256,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(d_alignments,attent_layer->d_total_hs_mat,d_h_t,LSTM_size,minibatch_size,attent_layer->d_batch_info);
	
	cudaMemsetAsync(d_normal_alignments,0,minibatch_size * attent_layer->longest_sent*sizeof(dType),attent_layer->layer_info.s0);
	
	//normalization
	normalization_alignment_kernel<<<1,minibatch_size,0,attent_layer->layer_info.s0>>>(d_normal_alignments,d_alignments,minibatch_size,attent_layer->d_batch_info);
	
	//compute c_t of attention_weight*d_total_hs_mat
	create_my_c_t_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_normal_alignments,attent_layer->d_total_hs_mat, d_c_t, LSTM_size,minibatch_size, attent_layer->d_batch_info);

	//W_c_p1 * c_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_c_p1,Embedding_size,
		d_c_t,LSTM_size,&beta,d_final_temp_1,Embedding_size),"attention forward p_t part 1\n");


	// //W_c_p2 * h_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_c_p2,Embedding_size,
		d_h_t_att,LSTM_size,&beta,d_final_temp_2,Embedding_size),"attention forward p_t part 2\n");

	//add in the bias and tanh
	tanh_att_forward_kernel<<<std::min(256,(Embedding_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_final_temp_2,d_final_temp_1,d_final_temp_2,attent_layer->d_output_bias,Embedding_size,minibatch_size);
	CUDA_GET_LAST_ERROR("attention tanh forward");

	zero_h_t<<<std::min(256,(Embedding_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_final_temp_2, *d_indicies_mask,Embedding_size,minibatch_size);

	//send h_tild to the lowest level
	//if last index, then there is nothing to copy to
	if(feed_input && index != (attent_layer->longest_sent-1)) {
		cudaMemcpyAsync(d_lower_htild,d_final_temp_2,Embedding_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,attent_layer->layer_info.s0); //d_lower_htild --> nodes[i+1].d_h_tild
	}

	cudaEventRecord(attent_layer->layer_info.forward_prop_done,attent_layer->layer_info.s0);
}


template<typename dType>
void attention_node<dType>::back_prop() {

	cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.start_backward,0);
	
	if(feed_input && attent_layer->transfer_done) {
		cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.error_htild_below,0);
		add_two_mats_kernel<<<std::min(256,(Embedding_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild,d_ERRtTOn_htild_below,Embedding_size*minibatch_size);
	}

	attent_layer->transfer_done = true; //for feed input errors, the first error we dont want to add

	dType alpha = 1;
	dType beta = 1;

	//test this for gradients
	zero_h_t<<<std::min(256,(Embedding_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild, *d_indicies_mask,Embedding_size,minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION zero h_t");

	//multiply the gradient coming down by 1-tanh()^2
	//d_d_ERRt_ht_tild = d_d_ERRt_ht_tild *e (1 - d_final_temp_2 *e d_final_temp_2)
	tanh_grad_kernel<<< std::min(256,(Embedding_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild,d_d_ERRt_ht_tild,d_final_temp_2,Embedding_size*minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION tanh grad");

	//calculate gradient with respect to W_c_1
	//d_W_c_p1_grad += d_d_ERRt_ht_tild * d_c_t.T
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,Embedding_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRt_ht_tild,Embedding_size,d_c_t,LSTM_size,&beta,attent_layer->d_W_c_p1_grad,Embedding_size),"Attention backprop W_c_1 grad\n");

	//calculate gradient with respect to W_c_2
	//d_W_c_p2_grad += d_d_ERRt_ht_tild * d_h_t_att.T
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,Embedding_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRt_ht_tild,Embedding_size,d_h_t_att,LSTM_size,&beta,attent_layer->d_W_c_p2_grad,Embedding_size),"Attention backprop W_c_2 grad\n");

	//calculate gradient with respect to output_bias
	//d_output_bias_grad += d_d_ERRt_ht_tild * d_ones_minibatch
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(attent_layer->handle,CUBLAS_OP_N,Embedding_size,minibatch_size,&alpha,d_d_ERRt_ht_tild,Embedding_size,
		attent_layer->d_ones_minibatch,1,&beta,attent_layer->d_output_bias_grad,1),"backprop b_i_grad failed\n");

	
	alpha = 1;
	beta = 0;
	
	//calculate error with respect to c_t
	//d_ERRnTOt_ct = d_W_c_p1.T * d_d_ERRt_ht_tild
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,
		&alpha,attent_layer->d_W_c_p1,Embedding_size,d_d_ERRt_ht_tild,Embedding_size,&beta,attent_layer->d_ERRnTOt_ct,LSTM_size),"Attention backprop d_ERRnTOt_ct\n");

	//calculate first part of error with respect to h_t
	//d_ERRnTOt_ht_p1 = d_W_c_p2.T * d_d_ERRt_ht_tild
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,
		&alpha,attent_layer->d_W_c_p2,Embedding_size,d_d_ERRt_ht_tild,Embedding_size,&beta,attent_layer->d_ERRnTOt_ht_p1,LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");

	
	
	cudaMemsetAsync(attent_layer->d_ERRnTOt_as,0,minibatch_size * attent_layer->longest_sent*sizeof(dType),attent_layer->layer_info.s0);
	//compute gradient for attention weight d_normal_alignments
	//d_ERRnTOt_as = sum(d_ERRnTOt_ct * d_total_hs_mat[j])
	my_error_alignments_kernel<<<256,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_as, attent_layer->d_ERRnTOt_ct, attent_layer->d_total_hs_mat, LSTM_size,minibatch_size, attent_layer->d_batch_info);
	CUDA_GET_LAST_ERROR("attention create ct");
	
	//compute gradient for h_total_hs_mat (out of attention)
	//d_total_hs_error[j] += d_ERRnTOt_ct * d_normal_alignments 
	my_error_hs_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ct, d_normal_alignments,attent_layer->d_total_hs_error,LSTM_size,minibatch_size,attent_layer->d_batch_info);
   
	cudaMemsetAsync(attent_layer->d_ERRnTOt_hsht,0,minibatch_size * attent_layer->longest_sent*sizeof(dType),attent_layer->layer_info.s0);
   	
	//softmax gradient calculation in attention weights
	//d_ERRnTOt_hsht
	my_error_alignments_to_hsht_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_as,attent_layer->d_ERRnTOt_hsht,d_normal_alignments,minibatch_size,attent_layer->d_batch_info);
	
	//compute gradient for h_total_hs_mat (in attention)
	//d_total_hs_error
	my_error_hsht_to_hs_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_ERRnTOt_hsht,d_h_t,LSTM_size,minibatch_size,attent_layer->d_batch_info);
	
	//compute gradient for h_t (in attention)
	//d_ERRnTOt_ht_p1
	my_error_hsht_to_ht_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ht_p1,attent_layer->d_ERRnTOt_hsht,attent_layer->d_total_hs_mat,LSTM_size,minibatch_size,attent_layer->d_batch_info);

	if(dropout) {
		dropout_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(d_dropout_mask,dropout_rate,attent_layer->d_ERRnTOt_ht_p1,LSTM_size*minibatch_size);
	}
	
	cudaMemcpyAsync(d_d_ERRt_ht_tild,attent_layer->d_ERRnTOt_ht_p1,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDeviceToDevice,attent_layer->layer_info.s0);
	CUDA_GET_LAST_ERROR("TRANSFER COPY ATTENTION");

	cudaEventRecord(attent_layer->layer_info.backward_prop_done,attent_layer->layer_info.s0);
}


