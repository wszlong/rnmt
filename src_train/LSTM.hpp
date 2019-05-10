
//Constructor
template<typename dType>
LSTM_IH_Node<dType>::LSTM_IH_Node(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m,int index,bool dropout,dType dropout_rate, bool bi_dir) 
{
	model = m;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;

	init_LSTM_GPU(Embedding_size,LSTM_size,minibatch_size,vocab_size,m);

	//model = m;
	this->minibatch_size = minibatch_size;
	this->Embedding_size = Embedding_size;
	this->LSTM_size = LSTM_size;
	this->index = index;
	this->bi_dir = bi_dir;
}


template<typename dType>
void LSTM_IH_Node<dType>::init_LSTM_GPU(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m) {

	cudaSetDevice(model->ih_layer_info.device_number);

	full_matrix_setup(&h_o_t,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_t,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_f_t,&d_f_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_prime_t_tanh,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_i_t,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_sparse_lookup,&d_sparse_lookup,Embedding_size,minibatch_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
	
	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	cudaMemset(d_d_ERRt_ht,0,LSTM_size*minibatch_size*sizeof(dType));

	if(dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, Embedding_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");	
	}
}

//this is to be called if feed input is true
template<typename dType>
void LSTM_IH_Node<dType>::attention_extra() {
	cudaSetDevice(model->ih_layer_info.device_number);
	dType *h_temp;
	full_matrix_setup(&h_temp,&d_ERRnTOt_h_tild,Embedding_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_h_tild,Embedding_size,minibatch_size); //
	feed_input = true;
}


template<typename dType>
void LSTM_IH_Node<dType>::update_vectors_forward_GPU(int *d_input_vocab_indices,int *d_input_vocab_indices_01,
	dType *d_h_t_prev,dType *d_c_t_prev,int current_length) 
{
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;
	this->d_input_vocab_indices = d_input_vocab_indices;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
	this->current_length = current_length;
}

template<typename dType>
void LSTM_IH_Node<dType>::forward_prop() {
	forward_prop_GPU();
}

template<typename dType>
void LSTM_IH_Node<dType>::forward_prop_GPU() {
	
	cudaSetDevice(model->ih_layer_info.device_number);
	int threads_per_block = 128;
	int num_block_E = (Embedding_size+threads_per_block-1)/threads_per_block; //LSTM_size -> Embedding_size
	dim3 kernel_E(minibatch_size,num_block_E,1);
	
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block; 
	dim3 kernel(minibatch_size,num_block,1);
	CUDA_GET_LAST_ERROR("PRE SPARSE");
	
	//d_sparse_lookup: Embedding_size * minibatch_size	
	sparse_lookup_kernel<<< kernel_E,threads_per_block,0,model->ih_layer_info.s0>>>(d_sparse_lookup,model->d_W,d_input_vocab_indices,minibatch_size,Embedding_size);
	CUDA_GET_LAST_ERROR("SPARSE");
	
	if(dropout && model->model->train) {

		if(!model->model->grad_check_flag) {
			curandSetStream(model->rand_gen, model->ih_layer_info.s0);
			curandGenerateUniform_wrapper(d_dropout_mask,Embedding_size*minibatch_size,model->rand_gen);
		}
		dropout_kernel<<<256,256,0,model->ih_layer_info.s0>>>(d_dropout_mask,model->dropout_rate,d_sparse_lookup,Embedding_size*minibatch_size);
	}
	cudaEventRecord(model->ih_layer_info.sparse_forward_start,model->ih_layer_info.s0);

	dType alpha = 1;
	dType beta = 0;

	//std::cout << "i_t start\n";
	//*******************i_t start**********************//
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s1);
	cudaStreamWaitEvent(model->ih_layer_info.s1,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_M_i,LSTM_size,
		d_sparse_lookup,Embedding_size,&beta,model->d_temp1,LSTM_size),"Forward prop i_t temp1 failed\n");
	cudaEventRecord(model->ih_layer_info.i_t_part1,model->ih_layer_info.s1);

	
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s2);
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hi,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp2,LSTM_size),"Forward prop i_t temp2 failed\n");


	if(feed_input && index!=0) {

		//std::cout << "FEED FORWARD 1\n";
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s2);
		cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.attention_forward,0);  //
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_Q_i,LSTM_size,
			d_h_tild,Embedding_size,&beta,model->d_temp9,LSTM_size),"Forward prop i_t temp2 failed\n");
	}


	CUDA_GET_LAST_ERROR("i_t for lower level LSTM P1");
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.i_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_b_i,LSTM_size);
	}
	else {
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_temp9,model->d_b_i,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("i_t for lower level LSTM");
	cudaEventRecord(model->ih_layer_info.i_t_full,model->ih_layer_info.s2);

	//**************f_t start***********************//
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s3);
	cudaStreamWaitEvent(model->ih_layer_info.s3,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_M_f,LSTM_size,
		d_sparse_lookup,Embedding_size,&beta,model->d_temp3,LSTM_size),"Forward prop f_t temp3 failed\n");
	cudaEventRecord(model->ih_layer_info.f_t_part1,model->ih_layer_info.s3);


	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s4);
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hf,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");


	if(feed_input && index!=0) {
		
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s4);
		cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_Q_f,LSTM_size,
			d_h_tild,Embedding_size,&beta,model->d_temp10,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.f_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_b_f,LSTM_size);
	}
	else {
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_temp10,model->d_b_f,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("f_t");
	cudaEventRecord(model->ih_layer_info.f_t_full,model->ih_layer_info.s4);

	//*************************c_prime_t_tanh start************************/
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
	cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_M_c,LSTM_size,
		d_sparse_lookup,Embedding_size,&beta,model->d_temp5,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");
	cudaEventRecord(model->ih_layer_info.c_prime_t_tanh_part1,model->ih_layer_info.s5);


	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hc,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");
	
	if(feed_input && index!=0) {
		
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_Q_c,LSTM_size,
			d_h_tild,Embedding_size,&beta,model->d_temp11,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.c_prime_t_tanh_part1,0);
	if(!feed_input || index==0) {
		forward_tanh_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_b_c,LSTM_size);
	}
	else {
		forward_tanh_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_temp11,model->d_b_c,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");
	cudaEventRecord(model->ih_layer_info.c_prime_t_tanh_full,model->ih_layer_info.s6);

	//***********************o_t start*************************//
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
	cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_M_o,LSTM_size,
		d_sparse_lookup,Embedding_size,&beta,model->d_temp7,LSTM_size),"Forward prop o_t temp1 failed\n");
	cudaEventRecord(model->ih_layer_info.o_t_part1,model->ih_layer_info.s7);


	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_ho,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp8,LSTM_size),"Forward prop o_t temp2 failed\n");


	if(feed_input && index!=0) {
		
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,model->d_Q_o,LSTM_size,
			d_h_tild,Embedding_size,&beta,model->d_temp12,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.o_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_b_o,LSTM_size);
	}
	else {
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_temp12,model->d_b_o,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("o_t");
	cudaEventRecord(model->ih_layer_info.o_t_full,model->ih_layer_info.s8);

	//*********************************c_t start********************************//
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.i_t_full,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.f_t_full,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.c_prime_t_tanh_full,0);
	forward_c_t_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_c_t,d_f_t,d_c_t_prev,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("c_t");

	//********************************h_t start*********************************//
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.o_t_full,0);
	forward_h_t_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");

	//
	zero_c_t_and_h_t<<< kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_c_t,d_input_vocab_indices_01,LSTM_size);
	CUDA_GET_LAST_ERROR("zero");

	send_h_t_above();
	
	cudaSetDevice(0);

}

template<typename dType>
void LSTM_IH_Node<dType>::send_h_t_above() {
	

	//send the finished h_t to the above layer
	//the multigpu synchronization structure
	
	if(bi_dir){ //bidirection encoder
		if(model->upper_layer.copy_h_t) {
			cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[current_length-1-index].d_h_t_below_bi, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
		}
		else {
			model->upper_layer.hidden_layer->nodes[current_length-1-index].d_h_t_below_bi = d_h_t;
		}
	}
	else{ //normal
		if(model->upper_layer.copy_h_t) {
			cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
		}
		else {
			model->upper_layer.hidden_layer->nodes[index].d_h_t_below = d_h_t;
		}
	}
	cudaEventRecord(model->ih_layer_info.h_t_below_transfer,model->ih_layer_info.s0);

}


template<typename dType>
void LSTM_IH_Node<dType>::backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct)//,dType *d_d_ERRt_ht) 
{
	this->d_d_ERRnTOtp1_ht = d_d_ERRnTOtp1_ht;
	this->d_d_ERRnTOtp1_ct = d_d_ERRnTOtp1_ct;
	//this->d_d_ERRt_ht = d_d_ERRt_ht;
}


template<typename dType>
void LSTM_IH_Node<dType>::back_prop_GPU(int index) {

	cudaSetDevice(model->ih_layer_info.device_number);

	cudaStreamWaitEvent(model->ih_layer_info.s0,model->upper_layer.hidden_layer->hh_layer_info.d_ERR_ht_done,0);
	

	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.htm1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.ctm1_done,0);

	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_grad_full_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hi_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hf_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_ho_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hc_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_i_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_f_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_o_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_c_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_i_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_f_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_o_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_c_grad_done,0);


	dType alpha = 1;
	dType beta = 1;
	
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	//d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ht,LSTM_size,
		&beta,d_d_ERRt_ht,LSTM_size,model->d_d_ERRnTOt_ht,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	int num_block_E = (Embedding_size+threads_per_block-1)/threads_per_block;
	dim3 kernel_E(minibatch_size,num_block_E,1);
	

	//d_d_ERRt_ct = d_d_ERRnTOt_ht * d_o_t * (1 - tanh(d_c_t)*tanh(d_c_t))
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(model->d_d_ERRt_ct,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");

	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,model->d_d_ERRt_ct,LSTM_size,model->d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");

	//zero out columns of d_ERRnTOt_ht and d_ERRnTOt_ct
	zero_columns_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ht,d_input_vocab_indices_01,model->d_d_ERRnTOt_ht);
	CUDA_GET_LAST_ERROR("BP h_tn");
	zero_columns_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ct,d_input_vocab_indices_01,model->d_d_ERRnTOt_ct);
	CUDA_GET_LAST_ERROR("BP c_tn");

	cudaEventRecord(model->ih_layer_info.backprop_init,model->ih_layer_info.s0);

	//d_d_ERRnTOt_ot = d_d_ERRnTOt_ht * tanh(d_c_t) * d_o_t * (1-d_o_t)
	cudaStreamWaitEvent(model->ih_layer_info.s1,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ot_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s1>>>(model->d_d_ERRnTOt_ot,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP o_tn");
	cudaEventRecord(model->ih_layer_info.err_ot_done,model->ih_layer_info.s1);
	
	//d_d_ERRnTOt_ft = d_d_ERRnTOt_ct * d_c_t_prev * d_f_t * (1-d_f_t)
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(model->d_d_ERRnTOt_ft,model->d_d_ERRnTOt_ct,d_c_t_prev,d_f_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");
	cudaEventRecord(model->ih_layer_info.err_ft_done,model->ih_layer_info.s2);

	//d_d_ERRnTOt_tanhcpt = d_d_ERRnTOt_ct * d_i_t * (1-d_c_prime_t_tanh*d_c_prime_t_tanh)
	cudaStreamWaitEvent(model->ih_layer_info.s3,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s3>>>(model->d_d_ERRnTOt_tanhcpt,model->d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
	cudaEventRecord(model->ih_layer_info.err_tanhcpt_done,model->ih_layer_info.s3);
		
	//d_d_ERRnTOt_it = d_d_ERRnTOt_ct * d_c_prime_t_tanh * d_i_t * (1-d_i_t)
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(model->d_d_ERRnTOt_it,model->d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");
	cudaEventRecord(model->ih_layer_info.err_it_done,model->ih_layer_info.s4);


	dType alpha2 = 1;
	dType beta2 = 0;
	
	//d_temp1 = d_W_ho.T * d_d_ERRnTOt_ot
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
	cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_ho,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p1_done,model->ih_layer_info.s5);
	
	//d_temp2 = d_W_hf.T * d_d_ERRnTOt_ft
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hf,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s6);
	
	//d_temp3 = d_W_hi.T * d_d_ERRnTOt_it
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
	cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hi,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s7);
	
	//d_temp4 = d_W_hc.T * d_d_ERRnTOt_tanhcpt
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hc,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s8);


	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p4_done,0);
	//d_d_ERRnTOt_htM1 = d_temp1 + d_temp2 + d_temp3 + d_temp4
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1");

	cudaEventRecord(model->ih_layer_info.htm1_done_temp,model->ih_layer_info.s9);


	//send error to the attention model
	if(feed_input && index!=0) {

		//d_temp1 = d_Q_o.T * d_d_ERRnTOt_ot
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
		cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,Embedding_size),"Error backprop temp1 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p1_done,model->ih_layer_info.s5);

		//d_temp2 = d_Q_f.T * d_d_ERRnTOt_ft
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,Embedding_size),"Error backprop temp2 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s6);

		//d_temp3 = d_Q_i.T * d_d_ERRnTOt_it
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
		cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,Embedding_size),"Error backprop temp3 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s7);

		//d_temp4 = d_Q_c.T * d_d_ERRnTOt_tanhcpt
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,Embedding_size),"Error backprop temp4 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s8);

		
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p1_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p2_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p3_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p4_done,0);
		///d_ERRnTOt_h_tild = d_temp1 + d_temp2 + d_temp3 + d_temp4
		add_four_matrices_kernel<<< kernel_E,threads_per_block,0,model->ih_layer_info.s9>>>(d_ERRnTOt_h_tild,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,Embedding_size);
		CUDA_GET_LAST_ERROR("BP h tilda");
		
		cudaMemcpyAsync(d_ERRnTOt_h_tild_cpy,d_ERRnTOt_h_tild,Embedding_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->ih_layer_info.s9);
		cudaEventRecord(model->ih_layer_info.error_htild_below,model->ih_layer_info.s9);
	
	}
	
	//dont record this event until after the feed input
	cudaEventRecord(model->ih_layer_info.htm1_done,model->ih_layer_info.s9);

	//d_d_ERRnTOt_ctM1 = d_d_ERRnTOt_ct * d_f_t
	cudaStreamWaitEvent(model->ih_layer_info.s10,model->ih_layer_info.backprop_init,0);
	elementwise_mult_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s10>>>(model->d_d_ERRnTOt_ct,d_f_t,model->d_d_ERRnTOt_ctM1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");

	cudaEventRecord(model->ih_layer_info.ctm1_done,model->ih_layer_info.s10);

	compute_gradients_GPU(); //!!

}



template<typename dType>
void LSTM_IH_Node<dType>::compute_gradients_GPU() {

	dType alpha = 1;
	dType beta = 1;

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s11);
	cudaStreamWaitEvent(model->ih_layer_info.s11,model->ih_layer_info.err_it_done,0);
	//d_W_hi_grad += d_d_ERRnTOt_it * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hi_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hi_grad_done,model->ih_layer_info.s11);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s12);
	cudaStreamWaitEvent(model->ih_layer_info.s12,model->ih_layer_info.err_ft_done,0);
	//d_W_hf_grad += d_d_ERRnTOt_ft * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hf_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hf_grad_done,model->ih_layer_info.s12);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s13);
	cudaStreamWaitEvent(model->ih_layer_info.s13,model->ih_layer_info.err_tanhcpt_done,0);
	//d_W_hc_grad += d_d_ERRnTOt_tanhcpt * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hc_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hc_grad_done,model->ih_layer_info.s13);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s14);
	cudaStreamWaitEvent(model->ih_layer_info.s14,model->ih_layer_info.err_ot_done,0);
	//d_W_ho_grad += d_d_ERRnTOt_ot * d_h_t_prev.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_ho_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_ho_grad_done,model->ih_layer_info.s14);


	alpha = 1;
	beta = 1;

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s15);
	cudaStreamWaitEvent(model->ih_layer_info.s15,model->ih_layer_info.err_it_done,0);
	//d_M_i_grad += d_d_ERRnTOt_it * d_sparse_lookup.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_sparse_lookup,Embedding_size,&beta,model->d_M_i_grad,LSTM_size),"Backprop M_i grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_i_grad_done,model->ih_layer_info.s15);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s16);
	cudaStreamWaitEvent(model->ih_layer_info.s16,model->ih_layer_info.err_ft_done,0);
	//d_M_f_grad += d_d_ERRnTOt_ft * d_sparse_lookup.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_sparse_lookup,Embedding_size,&beta,model->d_M_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_f_grad_done,model->ih_layer_info.s16);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s17);
	cudaStreamWaitEvent(model->ih_layer_info.s17,model->ih_layer_info.err_ot_done,0);
	//d_M_o_grad += d_d_ERRnTOt_ot * d_sparse_lookup.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_sparse_lookup,Embedding_size,&beta,model->d_M_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_o_grad_done,model->ih_layer_info.s17);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s18);
	cudaStreamWaitEvent(model->ih_layer_info.s18,model->ih_layer_info.err_tanhcpt_done,0);
	//d_M_c_grad += d_d_ERRnTOt_tanhcpt * d_sparse_lookup.T
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_sparse_lookup,Embedding_size,&beta,model->d_M_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_c_grad_done,model->ih_layer_info.s18);
	
	//cout<<"show d_M_c_grad: "<<endl;
	//show_matrix(model->d_M_c_grad, LSTM_size, Embedding_size);


	//resuse all events from M_i
	if(feed_input && index!=0) {
		alpha = 1;
		beta = 1;
		
		//d_Q_i_grad = d_d_ERRnTOt_it * d_h_tild.T
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s15);
		cudaStreamWaitEvent(model->ih_layer_info.s15,model->ih_layer_info.err_it_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_it,LSTM_size,d_h_tild,Embedding_size,&beta,model->d_Q_i_grad,LSTM_size),"Backprop Q_i grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_i_grad_done,model->ih_layer_info.s15);

		//d_Q_f_grad = d_d_ERRnTOt_ft * d_h_tild.T
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s16);
		cudaStreamWaitEvent(model->ih_layer_info.s16,model->ih_layer_info.err_ft_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ft,LSTM_size,d_h_tild,Embedding_size,&beta,model->d_Q_f_grad,LSTM_size),"Backprop Q_f grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_f_grad_done,model->ih_layer_info.s16);

		//d_Q_o_grad = d_d_ERRnTOt_ot * d_h_tild.T
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s17);
		cudaStreamWaitEvent(model->ih_layer_info.s17,model->ih_layer_info.err_ot_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ot,LSTM_size,d_h_tild,Embedding_size,&beta,model->d_Q_o_grad,LSTM_size),"Backprop Q_o grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_o_grad_done,model->ih_layer_info.s17);

		//d_Q_c_grad = d_d_ERRnTOt_tanhcpt * d_h_tild.T
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s18);
		cudaStreamWaitEvent(model->ih_layer_info.s18,model->ih_layer_info.err_tanhcpt_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,Embedding_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_tild,Embedding_size,&beta,model->d_Q_c_grad,LSTM_size),"Backprop Q_c grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_c_grad_done,model->ih_layer_info.s18);
	}

	
	//d_b_i_grad += d_d_ERRnTOt_it * d_ones_minibatch(vector)
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s19);
	cudaStreamWaitEvent(model->ih_layer_info.s19,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_it,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_i_grad,1),"backprop b_i_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_i_grad_done,model->ih_layer_info.s19);

	//d_b_f_grad += d_d_ERRnTOt_ft * d_ones_minibatch
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s20);
	cudaStreamWaitEvent(model->ih_layer_info.s20,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ft,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_f_grad,1),"backprop b_f_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_f_grad_done,model->ih_layer_info.s20);

	//d_b_o_grad += d_d_ERRnTOt_ot * d_ones_minibatch
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s21);
	cudaStreamWaitEvent(model->ih_layer_info.s21,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ot,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_o_grad,1),"backprop b_o_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_o_grad_done,model->ih_layer_info.s21);

	//d_b_c_grad += d_d_ERRnTOt_tanhcpt * d_ones_minibatch
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s22);
	cudaStreamWaitEvent(model->ih_layer_info.s22,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_tanhcpt,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_c_grad,1),"backprop b_c_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_c_grad_done,model->ih_layer_info.s22);
	

	//******************************update word ebedding********************************//
	alpha = 1;
	beta = 0;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s23);
	cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.err_it_done,0);
	//d_temp5 = d_M_i.T * d_d_ERRnTOt_it
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta,
		model->d_temp5,Embedding_size),"cublas W gradient failed temp5\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p1_done,model->ih_layer_info.s23);

	//d_temp6 = d_M_f.T * d_d_ERRnTOt_ft
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s24);
	cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta,
		model->d_temp6,Embedding_size),"cublas W gradient failed temp6\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p2_done,model->ih_layer_info.s24);

	//d_temp7 = d_M_o.T * d_d_ERRnTOt_ot
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s25);
	cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta,
		model->d_temp7,Embedding_size),"cublas W gradient failed temp7\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p3_done,model->ih_layer_info.s25);

	//d_temp8 = d_M_c.T * d_d_ERRnTOt_tanhcpt
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s26);
	cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta,
		model->d_temp8,Embedding_size),"cublas W gradient failed temp8\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p4_done,model->ih_layer_info.s26);

	
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p2_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p3_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p4_done,0);
	
	int threads_per_block = 128;
	int num_block = (Embedding_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);

	if(!dropout) {	
		//first sum = d_temp5 + d_temp6 + d_temp7 + d_temp8
		//then atomicAdd	
		W_small_gradient_kernel<<<256,256,0,model->ih_layer_info.s27>>>(model->d_small_W_grad,model->d_reverse_unique_indicies,model->d_temp5,
			model->d_temp6,model->d_temp7,model->d_temp8,d_input_vocab_indices,Embedding_size,minibatch_size);
	}
	else {
		W_small_dropout_gradient_kernel<<<256,256,0,model->ih_layer_info.s27>>>(model->d_small_W_grad,model->d_reverse_unique_indicies,model->d_temp5,
			model->d_temp6,model->d_temp7,model->d_temp8,d_input_vocab_indices,Embedding_size,minibatch_size,d_dropout_mask,dropout_rate);
	}
	CUDA_GET_LAST_ERROR("BP w_grad");

	//if( bi_dir && model->upper_layer.source_side) {
		

		//cout<<"show d_temp8: "<<endl;
		//show_matrix(model->d_temp8, LSTM_size, minibatch_size);
		
		//cout<<"show d_small_W_grad: "<<endl;
		//show_matrix(model->d_small_W_grad, LSTM_size, model->w_grad_len);
	//}

	cudaEventRecord(model->ih_layer_info.W_grad_full_done,model->ih_layer_info.s27);
}




