
//Constructor
template<typename dType>
LSTM_HH_Node<dType>::LSTM_HH_Node(int Embedding_size,int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m,int index,bool dropout,dType dropout_rate) {

	model = m;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;

	init_LSTM_GPU(Embedding_size,LSTM_size,minibatch_size,m);

	//model = m;
	this->minibatch_size = minibatch_size;
	this->Embedding_size = Embedding_size;
	this->LSTM_size = LSTM_size;
	this->index = index;
}

template<typename dType>
void LSTM_HH_Node<dType>::init_LSTM_GPU(int Embedding_size,int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m) {

	cudaSetDevice(model->hh_layer_info.device_number);

	full_matrix_setup(&h_o_t,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_t,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_f_t,&d_f_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_prime_t_tanh,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_i_t,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t_below,&d_h_t_below,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t_below,&d_h_t_below_bi,LSTM_size,minibatch_size); //
	
	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	cudaMemset(d_d_ERRt_ht,0,LSTM_size*minibatch_size*sizeof(dType));

	if(dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	}
}

template<typename dType>
void LSTM_HH_Node<dType>::update_vectors_forward_GPU(int *d_input_vocab_indices_01,
	dType *d_h_t_prev,dType *d_c_t_prev, int current_length) 
{
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
	this->current_length = current_length;
}

template<typename dType>
void LSTM_HH_Node<dType>::forward_prop_sync(cudaStream_t &my_s) {

	if(model->lower_layer.copy_d_Err_ht) {
		if(model->lower_layer.lower_input) {
			cudaStreamWaitEvent(my_s,model->lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
		}
		else {
			cudaStreamWaitEvent(my_s,model->lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
		}
	}
	cudaStreamWaitEvent(my_s,model->hh_layer_info.h_t_below_transfer,0);
	cudaStreamWaitEvent(my_s,model->hh_layer_info.dropout_done,0);
}


template<typename dType>
void LSTM_HH_Node<dType>::forward_prop_sync_bi(cudaStream_t &my_s) {

	if(model->lower_layer.copy_d_Err_ht) {
		if(model->lower_layer.lower_input) {
			cudaStreamWaitEvent(my_s,model->lower_layer.input_layer_bi->ih_layer_info.h_t_below_transfer,0);
		}
	}
	cudaStreamWaitEvent(my_s,model->hh_layer_info.h_t_below_transfer,0);
	cudaStreamWaitEvent(my_s,model->hh_layer_info.dropout_done,0);
}


template<typename dType>
void LSTM_HH_Node<dType>::forward_prop() {
	
	forward_prop_GPU();
}

template<typename dType>
void LSTM_HH_Node<dType>::forward_prop_GPU() {

	cudaSetDevice(model->hh_layer_info.device_number);
	
	if(dropout && model->model->train) {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.h_t_below_transfer,0);
		if(model->lower_layer.copy_d_Err_ht && model->lower_layer.lower_input) {
			cudaStreamWaitEvent(model->hh_layer_info.s0,model->lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
		}
		else {
			cudaStreamWaitEvent(model->hh_layer_info.s0,model->lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
		}

		if(!model->model->grad_check_flag) {
			curandSetStream(model->rand_gen,model->hh_layer_info.s0);
			curandGenerateUniform_wrapper(d_dropout_mask,LSTM_size*minibatch_size,model->rand_gen); 
		}
		dropout_kernel<<<256,256,0,model->hh_layer_info.s0>>>(d_dropout_mask,dropout_rate,d_h_t_below,LSTM_size*minibatch_size);
		dropout_kernel<<<256,256,0,model->hh_layer_info.s0>>>(d_dropout_mask,dropout_rate,d_h_t_below_bi,LSTM_size*minibatch_size);
		cudaEventRecord(model->hh_layer_info.dropout_done,model->hh_layer_info.s0);
	}

	dType alpha =1;
	dType beta = 0;

	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);

	CUDA_GET_LAST_ERROR("CHECKPOINT 0");

	bool flag = true;

	//********************************i_t start******************************//
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s1);
	forward_prop_sync(model->hh_layer_info.s1);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_i,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp1,LSTM_size),"Forward prop i_t temp1 failed\n");
	
	//bidirection
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		//forward_prop_sync_bi(model->hh_layer_info.s1);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_U_i,LSTM_size,
			d_h_t_below_bi,LSTM_size,&beta,model->d_temp1_bi,LSTM_size),"Forward prop i_t temp1 failed\n");
	}
	cudaEventRecord(model->hh_layer_info.i_t_part1,model->hh_layer_info.s1);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s2);
	forward_prop_sync(model->hh_layer_info.s2);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hi,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp2,LSTM_size),"Forward prop i_t temp2 failed\n");


	cudaStreamWaitEvent(model->hh_layer_info.s2,model->hh_layer_info.i_t_part1,0);
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		forward_sigmoid_kernel_bi<<<kernel,threads_per_block,0,model->hh_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp1_bi,model->d_temp2,model->d_b_i,LSTM_size);
	}
	else {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_b_i,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("i_t LSTM HH");
	cudaEventRecord(model->hh_layer_info.i_t_full,model->hh_layer_info.s2);

	//**********************************f_t start****************************//
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s3);
	forward_prop_sync(model->hh_layer_info.s3);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_f,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp3,LSTM_size),"Forward prop f_t temp3 failed\n");
	//bi
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		//forward_prop_sync_bi(model->hh_layer_info.s3);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_U_f,LSTM_size,
			d_h_t_below_bi,LSTM_size,&beta,model->d_temp3_bi,LSTM_size),"Forward prop f_t temp3 failed\n");
	}
	cudaEventRecord(model->hh_layer_info.f_t_part1,model->hh_layer_info.s3);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s4);
	forward_prop_sync(model->hh_layer_info.s4);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hf,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");


	cudaStreamWaitEvent(model->hh_layer_info.s4,model->hh_layer_info.f_t_part1,0);
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		forward_sigmoid_kernel_bi<<<kernel,threads_per_block,0,model->hh_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp3_bi,model->d_temp4,model->d_b_f,LSTM_size);
	}
	else {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_b_f,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("f_t");
	cudaEventRecord(model->hh_layer_info.f_t_full,model->hh_layer_info.s4);
	
	//*****************************c_prime_t_tanh start**************************//
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	forward_prop_sync(model->hh_layer_info.s5);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_c,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp5,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");
	
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		//forward_prop_sync_bi(model->hh_layer_info.s5);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_U_c,LSTM_size,
			d_h_t_below_bi,LSTM_size,&beta,model->d_temp5_bi,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");
	}
	cudaEventRecord(model->hh_layer_info.c_prime_t_tanh_part1,model->hh_layer_info.s5);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	forward_prop_sync(model->hh_layer_info.s6);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hc,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");


	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.c_prime_t_tanh_part1,0);
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		forward_tanh_kernel_bi<<<kernel,threads_per_block,0,model->hh_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp5_bi,model->d_temp6,model->d_b_c,LSTM_size);
	}
	else {
		forward_tanh_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_b_c,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");
	cudaEventRecord(model->hh_layer_info.c_prime_t_tanh_full,model->hh_layer_info.s6);

	//********************************o_t start***********************************//
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	forward_prop_sync(model->hh_layer_info.s7);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_o,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp7,LSTM_size),"Forward prop o_t temp1 failed\n");
	
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		//forward_prop_sync_bi(model->hh_layer_info.s7);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_U_o,LSTM_size,
			d_h_t_below_bi,LSTM_size,&beta,model->d_temp7_bi,LSTM_size),"Forward prop o_t temp1 failed\n");
	}
	cudaEventRecord(model->hh_layer_info.o_t_part1,model->hh_layer_info.s7);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	forward_prop_sync(model->hh_layer_info.s8);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_ho,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp8,LSTM_size),"Forward prop o_t temp2 failed ZZZZZZZ\n");


	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.o_t_part1,0);
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {
		forward_sigmoid_kernel_bi<<<kernel,threads_per_block,0,model->hh_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp7_bi,model->d_temp8,model->d_b_o,LSTM_size);
	}
	else {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_b_o,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("o_t");
	cudaEventRecord(model->hh_layer_info.o_t_full,model->hh_layer_info.s8);

	//*********************************c_t start***************************//
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.i_t_full,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.f_t_full,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.c_prime_t_tanh_full,0);
	forward_c_t_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_c_t,d_f_t,d_c_t_prev,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("c_t");

	//*********************************h_t start***************************//
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.o_t_full,0);
	forward_h_t_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");
		

	zero_c_t_and_h_t<<< kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_h_t,d_c_t,d_input_vocab_indices_01,LSTM_size);
	CUDA_GET_LAST_ERROR("zero");

	send_h_t_above();
}


template<typename dType>
void LSTM_HH_Node<dType>::send_h_t_above() {

	//run forward prop for attention model
	if(attention_model) {
		cudaEventRecord(model->attent_layer->layer_info.start_forward,model->hh_layer_info.s0);
		model->attent_layer->nodes[index].forward_prop();
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->attent_layer->layer_info.forward_prop_done,0);	
	}

	if(model->upper_layer.copy_h_t) { //diffetent GPU
		//upper layer is softmax
		if(model->upper_layer.upper_softmax) {
			if(!model->upper_layer.source_side) { 
				cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), model->attent_layer->nodes[index].d_final_temp_2,Embedding_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
			}
		}
		//upper layer is hidden_layer
		else { 
			cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
		}
	}
	else { //same GPU
		if(model->upper_layer.upper_softmax) {
			if(!model->upper_layer.source_side) {
				model->upper_layer.softmax->set_ht_ptr(index,model->attent_layer->nodes[index].d_final_temp_2);
			}	
		}
		else {
			model->upper_layer.hidden_layer->nodes[index].d_h_t_below = d_h_t;
		}
	}
	
	cudaEventRecord(model->hh_layer_info.h_t_below_transfer,model->hh_layer_info.s0);
}


template<typename dType>
void LSTM_HH_Node<dType>::backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct) {
	this->d_d_ERRnTOtp1_ht = d_d_ERRnTOtp1_ht;
	this->d_d_ERRnTOtp1_ct = d_d_ERRnTOtp1_ct;
}

template<typename dType>
void LSTM_HH_Node<dType>::back_prop_GPU(int index) {

	bool flag = true;

	cudaSetDevice(model->hh_layer_info.device_number);

	if(model->upper_layer.upper_softmax) {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->upper_layer.softmax->get_ERR_ht_event(),0);
	}
	else {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->upper_layer.hidden_layer->hh_layer_info.htm1_done,0);
	}

	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.htm1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.ctm1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hi_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hf_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_ho_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hc_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_i_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_f_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_o_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_c_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_i_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_f_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_o_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_c_grad_done,0);

	if(attention_model) {
		
		cudaEventRecord(model->attent_layer->layer_info.start_backward,model->hh_layer_info.s0);
		model->attent_layer->nodes[index].back_prop();
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->attent_layer->layer_info.backward_prop_done,0);
	}

	//d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;
	dType alpha = 1;
	dType beta = 1;
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ht,LSTM_size,
		&beta,d_d_ERRt_ht,LSTM_size,model->d_d_ERRnTOt_ht,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

	//d_d_ERRt_ct = d_d_ERRnTOt_ht * d_o_t * (1 - tanh(d_c_t)*tanh(d_c_t))
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(model->d_d_ERRt_ct,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");

	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,model->d_d_ERRt_ct,LSTM_size,model->d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");

	//zero out columns of d_ERRnTOt_ht and d_ERRnTOt_ct
	zero_columns_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ht,d_input_vocab_indices_01,model->d_d_ERRnTOt_ht);
	CUDA_GET_LAST_ERROR("BP h_tn");
	zero_columns_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ct,d_input_vocab_indices_01,model->d_d_ERRnTOt_ct);
	CUDA_GET_LAST_ERROR("BP c_tn");

	cudaEventRecord(model->hh_layer_info.backprop_init,model->hh_layer_info.s0);

	//d_d_ERRnTOt_ot = d_d_ERRnTOt_ht * tanh(d_c_t) * d_o_t * (1-d_o_t)
	cudaStreamWaitEvent(model->hh_layer_info.s1,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ot_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s1>>>(model->d_d_ERRnTOt_ot,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP o_tn");
	cudaEventRecord(model->hh_layer_info.err_ot_done,model->hh_layer_info.s1);
	
	//d_d_ERRnTOt_ft = d_d_ERRnTOt_ct * d_c_t_prev * d_f_t * (1-d_f_t)
	cudaStreamWaitEvent(model->hh_layer_info.s2,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s2>>>(model->d_d_ERRnTOt_ft,model->d_d_ERRnTOt_ct,d_c_t_prev,d_f_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");
	cudaEventRecord(model->hh_layer_info.err_ft_done,model->hh_layer_info.s2);

	//d_d_ERRnTOt_tanhcpt = d_d_ERRnTOt_ct * d_i_t * (1-d_c_prime_t_tanh*d_c_prime_t_tanh)
	cudaStreamWaitEvent(model->hh_layer_info.s3,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s3>>>(model->d_d_ERRnTOt_tanhcpt,model->d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
	cudaEventRecord(model->hh_layer_info.err_tanhcpt_done,model->hh_layer_info.s3);
		
	//d_d_ERRnTOt_it = d_d_ERRnTOt_ct * d_c_prime_t_tanh * d_i_t * (1-d_i_t)	
	cudaStreamWaitEvent(model->hh_layer_info.s4,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s4>>>(model->d_d_ERRnTOt_it,model->d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");
	cudaEventRecord(model->hh_layer_info.err_it_done,model->hh_layer_info.s4);


	dType alpha2 = 1;
	dType beta2 = 0;

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	cudaStreamWaitEvent(model->hh_layer_info.s5,model->hh_layer_info.err_ot_done,0);
	//d_temp1 = d_M_o.T * d_d_ERRnTOt_ot
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	//d_temp1_bi
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_U_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1_bi,LSTM_size),"Error backprop temp1 htM1\n");
	}
	cudaEventRecord(model->hh_layer_info.htm1_p1_done,model->hh_layer_info.s5);
	

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.err_ft_done,0);
	//d_temp2 = d_M_f.T * d_d_ERRnTOt_ft
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	//d_temp3_bi
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_U_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp3_bi,LSTM_size),"Error backprop temp2 htM1\n");
	}
	cudaEventRecord(model->hh_layer_info.htm1_p2_done,model->hh_layer_info.s6);
	

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	cudaStreamWaitEvent(model->hh_layer_info.s7,model->hh_layer_info.err_it_done,0);
	//d_temp3 = d_M_i * d_d_ERRnTOt_it	
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	//d_temp5_bi
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_U_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp5_bi,LSTM_size),"Error backprop temp3 htM1\n");
	}
	cudaEventRecord(model->hh_layer_info.htm1_p3_done,model->hh_layer_info.s7);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.err_tanhcpt_done,0);
	//d_temp4 = d_M_c * d_d_ERRnTOt_tanhcpt
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	//d_temp7_bi
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_U_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp7_bi,LSTM_size),"Error backprop temp4 htM1\n");
	}
	cudaEventRecord(model->hh_layer_info.htm1_p4_done,model->hh_layer_info.s8);

	
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p4_done,0);
	//d_d_ERRnTOt_h_Below
	//d_final[index] = d_mat1[index] + d_mat2[index] + d_mat3[index] + d_mat4[index];
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_h_Below,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	
	//bidirection
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		add_four_matrices_kernel<<< kernel,threads_per_block,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_h_Below_bi,model->d_temp1_bi,model->d_temp3_bi,model->d_temp5_bi,model->d_temp7_bi,LSTM_size);
	}

	CUDA_GET_LAST_ERROR("BP htm1 below");


	if(dropout) {
		dropout_kernel<<<256,256,0,model->hh_layer_info.s9>>>(d_dropout_mask,dropout_rate,model->d_d_ERRnTOt_h_Below,LSTM_size*minibatch_size);
		
		if(model->lower_layer.lower_input && model->upper_layer.source_side) {	
			dropout_kernel<<<256,256,0,model->hh_layer_info.s9>>>(d_dropout_mask,dropout_rate,model->d_d_ERRnTOt_h_Below_bi,LSTM_size*minibatch_size);
		}
	}
	
	if(model->lower_layer.copy_d_Err_ht) {
		if(model->lower_layer.lower_input) {
			cudaMemcpyAsync(model->lower_layer.input_layer->nodes[index].d_d_ERRt_ht, model->d_d_ERRnTOt_h_Below, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s9);
			if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
				cudaMemcpyAsync(model->lower_layer.input_layer_bi->nodes[current_length-1-index].d_d_ERRt_ht, model->d_d_ERRnTOt_h_Below_bi, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s9);
			}
		}
		else {
			cudaMemcpyAsync(model->lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht, model->d_d_ERRnTOt_h_Below, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s9);
		}
	}
	else {
		if(model->lower_layer.lower_input) {
			model->lower_layer.input_layer->nodes[index].d_d_ERRt_ht = model->d_d_ERRnTOt_h_Below;
			if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
				model->lower_layer.input_layer_bi->nodes[current_length-1-index].d_d_ERRt_ht = model->d_d_ERRnTOt_h_Below_bi;
			}
		}
		else {
			model->lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht = model->d_d_ERRnTOt_h_Below;
		}
	}
	cudaEventRecord(model->hh_layer_info.d_ERR_ht_done,model->hh_layer_info.s9);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	cudaStreamWaitEvent(model->hh_layer_info.s5,model->hh_layer_info.d_ERR_ht_done,0);
	//d_temp1 = d_W_ho * d_d_ERRnTOt_ot
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_ho,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p1_done,model->hh_layer_info.s5);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.d_ERR_ht_done,0);
	//d_temp2 = d_W_hf * d_d_ERRnTOt_ft
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hf,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p2_done,model->hh_layer_info.s6);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	cudaStreamWaitEvent(model->hh_layer_info.s7,model->hh_layer_info.d_ERR_ht_done,0);
	//d_temp3 = d_W_hi * d_d_ERRnTOt_it
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hi,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p3_done,model->hh_layer_info.s7);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.d_ERR_ht_done,0);
	//d_temp4 = d_W_hc * d_d_ERRnTOt_tanhcpt
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hc,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p4_done,model->hh_layer_info.s8);


	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p4_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.d_ERR_ht_done,0);
	//d_d_ERRnTOt_htM1 = d_temp1 + d_temp2 + d_temp3 + d_temp4
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1");

	cudaEventRecord(model->hh_layer_info.htm1_done,model->hh_layer_info.s9);

	
	cudaStreamWaitEvent(model->hh_layer_info.s10,model->hh_layer_info.backprop_init,0);
	//d_d_ERRnTOt_ctM1 = d_d_ERRnTOt_ct ele* d_f_t 
	elementwise_mult_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s10>>>(model->d_d_ERRnTOt_ct,d_f_t,model->d_d_ERRnTOt_ctM1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");

	cudaEventRecord(model->hh_layer_info.ctm1_done,model->hh_layer_info.s10);

	compute_gradients_GPU(); //!!

	cudaSetDevice(0);
}


template<typename dType>
void LSTM_HH_Node<dType>::compute_gradients_GPU() {

	bool flag = true;

	dType alpha = 1;
	dType beta = 1;

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s11);
	cudaStreamWaitEvent(model->hh_layer_info.s11,model->hh_layer_info.err_it_done,0);
	//d_W_hi_grad += d_d_ERRnTOt_it * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hi_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hi_grad_done,model->hh_layer_info.s11);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s12);
	cudaStreamWaitEvent(model->hh_layer_info.s12,model->hh_layer_info.err_ft_done,0);
	//d_W_hf_grad += d_d_ERRnTOt_ft * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hf_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hf_grad_done,model->hh_layer_info.s12);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s13);
	cudaStreamWaitEvent(model->hh_layer_info.s13,model->hh_layer_info.err_tanhcpt_done,0);
	//d_W_hc_grad += d_d_ERRnTOt_tanhcpt * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hc_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hc_grad_done,model->hh_layer_info.s13);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s14);
	cudaStreamWaitEvent(model->hh_layer_info.s14,model->hh_layer_info.err_ot_done,0);
	//d_W_ho_grad += d_d_ERRnTOt_ot * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_ho_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_ho_grad_done,model->hh_layer_info.s14);


	alpha = 1;
	beta = 1;

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s15);
	cudaStreamWaitEvent(model->hh_layer_info.s15,model->hh_layer_info.err_it_done,0);
	//d_M_i_grad += d_d_ERRnTOt_it * d_h_t_below.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_i_grad,LSTM_size),"Backprop M_i grad cublas gemm failed\n");
	//d_U_i_grad += d_d_ERRnTOt_it * d_h_t_below_bi.T
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_it,LSTM_size,d_h_t_below_bi,LSTM_size,&beta,model->d_U_i_grad,LSTM_size),"Backprop M_i grad cublas gemm failed\n");	
	}
	cudaEventRecord(model->hh_layer_info.M_i_grad_done,model->hh_layer_info.s15);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s16);
	cudaStreamWaitEvent(model->hh_layer_info.s16,model->hh_layer_info.err_ft_done,0);
	//d_M_f_grad += d_d_ERRnTOt_ft * d_h_t_below.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	//
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_below_bi,LSTM_size,&beta,model->d_U_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	}
	cudaEventRecord(model->hh_layer_info.M_f_grad_done,model->hh_layer_info.s16);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s17);
	cudaStreamWaitEvent(model->hh_layer_info.s17,model->hh_layer_info.err_ot_done,0);
	//d_M_o_grad += d_d_ERRnTOt_ot * d_h_t_below.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	//
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_below_bi,LSTM_size,&beta,model->d_U_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	}
	cudaEventRecord(model->hh_layer_info.M_o_grad_done,model->hh_layer_info.s17);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s18);
	cudaStreamWaitEvent(model->hh_layer_info.s18,model->hh_layer_info.err_tanhcpt_done,0);
	//d_M_c_grad += d_d_ERRnTOt_tanhcpt * d_h_t_below.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	//
	if(model->lower_layer.lower_input && model->upper_layer.source_side && flag) {	
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_below_bi,LSTM_size,&beta,model->d_U_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	}
	cudaEventRecord(model->hh_layer_info.M_c_grad_done,model->hh_layer_info.s18);


	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s19);
	cudaStreamWaitEvent(model->hh_layer_info.s19,model->hh_layer_info.err_it_done,0);
	//d_b_i_grad += 1 * d_d_ERRnTOt_it * d_ones_minibatch（vector）
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_it,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_i_grad,1),"backprop b_i_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_i_grad_done,model->hh_layer_info.s19);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s20);
	cudaStreamWaitEvent(model->hh_layer_info.s20,model->hh_layer_info.err_ft_done,0);
	//d_b_f_grad += d_d_ERRnTOt_ft * d_ones_minibatch
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ft,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_f_grad,1),"backprop b_f_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_f_grad_done,model->hh_layer_info.s20);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s21);
	cudaStreamWaitEvent(model->hh_layer_info.s21,model->hh_layer_info.err_ot_done,0);
	//d_b_o_grad += d_d_ERRnTOt_ot * d_ones_minibatch
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ot,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_o_grad,1),"backprop b_o_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_o_grad_done,model->hh_layer_info.s21);

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s22);
	cudaStreamWaitEvent(model->hh_layer_info.s22,model->hh_layer_info.err_tanhcpt_done,0);
	//d_b_c_grad += d_d_ERRnTOt_tanhcpt * d_ones_minibatch
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_tanhcpt,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_c_grad,1),"backprop b_c_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_c_grad_done,model->hh_layer_info.s22);

}


