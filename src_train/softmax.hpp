
template<typename dType>
void softmax_layer<dType>::init_loss_layer(struct neuralMT_model<precision> *model,global_params &params)
{
	this->output_vocab_size = params.target_vocab_size;
	this->Embedding_size = params.Embedding_size;
	this->LSTM_size = params.LSTM_size;
	this->clip_gradients = params.clip_gradient;
	this->model = model;
	this->norm_clip = params.norm_clip;
	this->minibatch_size = params.minibatch_size;
	this->learning_rate = params.learning_rate;
	this->train_perplexity = params.train_perplexity;
	this->dropout = params.dropout;
	this->dropout_rate = params.dropout_rate;
	//adam
	this->ADAM = params.ADAM;
	this->alpha_adam = params.alpha_adam;
	this->beta_1 = params.beta_1;
	this->beta_2 = params.beta_2;
	this->epsilon = params.epsilon;

	init_softmax_layer_GPU(output_vocab_size,minibatch_size,model,params.norm_clip,params.Embedding_size,params.LSTM_size, clip_gradients,learning_rate,params.longest_sent);
}


template<typename dType>
void softmax_layer<dType>::init_softmax_layer_GPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int Embedding_size,int LSTM_size, bool clip_gradients,dType learning_rate,int longest_sent) {

	cudaSetDevice(s_layer_info.device_number);
	
	//for get_perplexity_GPU
	thrust_h_outputdist.resize(output_vocab_size * minibatch_size);
	thrust_h_normalization.resize(1 * minibatch_size);
	thrust_d_outputdist.resize(output_vocab_size * minibatch_size);
	thrust_d_normalization.resize(1 * minibatch_size);
	
	initialize_thrust_vector(thrust_h_outputdist,output_vocab_size * minibatch_size);
    initialize_thrust_vector(thrust_h_normalization,1 * minibatch_size);

    thrust_d_outputdist = thrust_h_outputdist;
    thrust_d_normalization = thrust_h_normalization;
    d_outputdist = thrust::raw_pointer_cast(&thrust_d_outputdist[0]);
    d_normalization = thrust::raw_pointer_cast(&thrust_d_normalization[0]);

    
	full_matrix_setup(&h_D,&d_D,output_vocab_size,Embedding_size);
	full_matrix_setup(&h_h_t,&d_h_t,Embedding_size,minibatch_size);
	full_matrix_setup(&h_b_d,&d_b_d,output_vocab_size,1);
	full_vector_setup_ones(&h_ones,&d_ones,output_vocab_size);//?
	
	//adam
	if(ADAM) {	
		full_matrix_setup_0(&h_D,&d_D_mt,output_vocab_size,Embedding_size);
		full_matrix_setup_0(&h_D,&d_D_vt,output_vocab_size,Embedding_size);
		full_matrix_setup_0(&h_b_d,&d_b_d_mt,output_vocab_size,1);
		full_matrix_setup_0(&h_b_d,&d_b_d_vt,output_vocab_size,1);
	}


	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,Embedding_size,minibatch_size);
	full_matrix_setup(&h_D_grad,&d_D_grad,output_vocab_size,Embedding_size);
	full_vector_setup(&h_b_d_grad,&d_b_d_grad,output_vocab_size);
	
	thrust_d_D_grad = thrust::device_pointer_cast(d_D_grad);
	thrust_d_b_d_grad = thrust::device_pointer_cast(d_b_d_grad);
	
	full_matrix_setup_0(&h_output_vocab_indices,&d_output_vocab_indices,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_output_vocab_indices_01,&d_output_vocab_indices_01,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_output_vocab_indices_01_float,&d_output_vocab_indices_01_float,minibatch_size,longest_sent);


	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n"); 
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n"); 
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist_perp, output_vocab_size*minibatch_size*sizeof(double)),"GPU memory allocation failed\n"); //?
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_train_perplexity, 1*sizeof(double)),"GPU memory allocation failed\n");
	cudaMemset(d_train_perplexity,0,1*sizeof(double));


	curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);
	boost::uniform_int<> unif_boost( 1, 1000000 );
	curandSetPseudoRandomGeneratorSeed(rand_gen,MY_CUDA::curr_seed);
	MY_CUDA::curr_seed+=7;


	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( softmax_node<dType>(Embedding_size,minibatch_size,output_vocab_size,i,dropout) );
	}

	//now start clearning matrices at the end of the minibatch instead of beginning
	cudaSetDevice(s_layer_info.device_number);

	clear_gradients();
	cudaSetDevice(0);
}

template<typename dType>
void softmax_layer<dType>::init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer) {
	lower_layer.init_lower_transfer_layer(lower_input,copy_d_Err_ht,input_layer,hidden_layer);
}

template<typename dType>
softmax_layer_gpu_info softmax_layer<dType>::gpu_init(int device_number) {
	s_layer_info.init(device_number);
	return s_layer_info;
}

template<typename dType>
void softmax_layer<dType>::prep_GPU_vocab_indices(int *h_output_vocab_indices_target,int current_length) {
	cudaSetDevice(s_layer_info.device_number);

	cudaMemcpy(d_output_vocab_indices, h_output_vocab_indices_target, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();

	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid=128;
	vocab_softmax<<<blocks_per_grid,threads_per_block>>>(d_output_vocab_indices,d_output_vocab_indices_01,d_output_vocab_indices_01_float,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("softmax perp");
	
	//cout<<"show prep_GPU_vocab_indices d_output_vocab_indices: "<<endl;
	//show_matrix_int(d_output_vocab_indices, minibatch_size, current_length);
		
	//cout<<"show prep_GPU_vocab_indices d_output_vocab_indices_01: "<<endl;
	//show_matrix_int(d_output_vocab_indices_01, minibatch_size, current_length);

	cudaSetDevice(0);
}


//Note d_h_t may lie on different GPU ?
//WARNING NEED A DEVICE SYNCHRONIZE WHEN DOING MULTIGPU
template<typename dType>
void softmax_layer<dType>::backprop_prep_GPU(dType *d_h_t,int step) 
{
	this->d_h_t = d_h_t;
	this->d_output_vocab_indices_single = d_output_vocab_indices + step;
	
	//cout<<"backprop_prep_GPU: "<<step<<" "<<"d_output_vocab_indices_single"<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size,10);
	
	this->d_output_vocab_indices_01_single = d_output_vocab_indices_01 + step;
	this->d_output_vocab_indices_01_float_single = d_output_vocab_indices_01_float + step;
}

template<typename dType>
void softmax_layer<dType>::backprop_prep_GPU_mgpu(int step) 
{
	this->d_output_vocab_indices_single = d_output_vocab_indices + step;
	
	//cout<<"backprop_prep_GPU_mgpu: "<<step<<" "<<"d_output_vocab_indices_single"<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size,10);
	
	this->d_output_vocab_indices_01_single = d_output_vocab_indices_01 + step;
	this->d_output_vocab_indices_01_float_single = d_output_vocab_indices_01_float + step;
}

template<typename dType>
void softmax_layer<dType>::forward_prop(int index) {

	forward_prop_GPU(index);

}

template<typename dType>
void softmax_layer<dType>::forward_prop_GPU(int index) {

	cudaSetDevice(s_layer_info.device_number);
	cudaStreamWaitEvent(s_layer_info.s0,lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);

	get_distribution_GPU(output_vocab_size,nodes[index].d_outputdist,d_D,d_b_d,nodes[index].d_h_t);
}

template<typename dType>
void softmax_layer<dType>::get_distribution_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D,dType *d_b_d,dType *d_h_t) 
{
	cudaSetDevice(s_layer_info.device_number);
	////////////////////////
	
	//cout<< "show d_h_t: "<<endl;
	//show_matrix(d_h_t, Embedding_size, minibatch_size);	
	
	//cout<< "show d_D: "<<endl;
	//show_matrix(d_D, output_vocab_size, Embedding_size);	

	//multiply the D matrix with the hidden state matrix
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,
	 output_vocab_size, minibatch_size, Embedding_size, &alpha, d_D, output_vocab_size,
	  d_h_t, Embedding_size, &beta, d_outputdist, output_vocab_size),"get_distribution cuBLAS call failed 1\n");


	//add the bias vector to the matrix
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_bias_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s0 >>>(output_vocab_size,d_outputdist,d_b_d,d_outputdist);
	CUDA_GET_LAST_ERROR("matrix_bias_kernel");

	outputdist_overflow_prevention_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist, d_outputdist, output_vocab_size);
	CUDA_GET_LAST_ERROR("outputdist_overflow_prevention_kernel");

	//cout<<"show d_outputdist: "<<endl;
	//show_matrix(d_outputdist, output_vocab_size, minibatch_size);
	
	//string test;
	//cin>>test;

	if(train_perplexity) {
		train_perplexity_kernel<<<1,1,0,s_layer_info.s0>>>(d_output_vocab_indices_single,d_output_vocab_indices_01_single,d_outputdist,
			d_train_perplexity,minibatch_size,output_vocab_size); 
	}

	cudaEventRecord(s_layer_info.outputdist_done,s_layer_info.s0);

}

template<typename dType>
dType *softmax_layer<dType>::get_ht_ptr(int index) {
	return nodes[index].d_h_t;
}

template<typename dType>
void softmax_layer<dType>::set_ht_ptr(int index,dType *d_h_t) {
	nodes[index].d_h_t = d_h_t;
}


//only pass back the error, not D or b_d gradients
template<typename dType>
void softmax_layer<dType>::back_prop1(int index) {

	back_prop1_GPU(index);
}

template<typename dType>
void softmax_layer<dType>::back_prop1_GPU(int index) {
	get_h_t_gradient_GPU(output_vocab_size,d_D,nodes[index].d_outputdist,nodes[index].d_d_ERRt_ht,index);
}


// update d_D and d_b
template<typename dType>
void softmax_layer<dType>::back_prop2(int index) {
	back_prop2_GPU(index);
}

template<typename dType>
void softmax_layer<dType>::back_prop2_GPU(int index) {
	compute_D_gradient_GPU(output_vocab_size,nodes[index].d_outputdist,d_D_grad,nodes[index].d_h_t);
	compute_b_d_gradient_GPU(output_vocab_size,nodes[index].d_outputdist,d_b_d_grad);
}


//get the error for the softmax with respect to h_t
template<typename dType>
void softmax_layer<dType>::get_h_t_gradient_GPU(int output_vocab_size,dType *d_D,dType *d_outputdist,dType *d_d_ERRt_ht,int index) {

	cudaSetDevice(s_layer_info.device_number);

	cudaStreamWaitEvent(s_layer_info.s1,s_layer_info.outputdist_done,0);
	dType alpha = -1;
	dType beta = 0;
	//multiply outputdist by D
	//d_d_ERRt_ht = -1 * d_D(T) * d_outputdist
	cublasSetStream(s_layer_info.handle,s_layer_info.s1);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,Embedding_size,minibatch_size,output_vocab_size,
		&alpha,d_D,output_vocab_size,d_outputdist,output_vocab_size,&beta,d_d_ERRt_ht,Embedding_size),"cuBLAS h_t gradient failed");

	//add in the D rows
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_row_to_matrix_column_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s1 >>>(d_d_ERRt_ht,d_d_ERRt_ht,d_D,d_output_vocab_indices_single,Embedding_size,output_vocab_size);
	CUDA_GET_LAST_ERROR("backward add D");

	//zero out columns
	int num_block_2 = (Embedding_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim_2(minibatch_size,num_block_2,1);
	zero_columns_kernel_128<<<kernel_dim_2,threads_per_block,0,s_layer_info.s1 >>>(Embedding_size,d_d_ERRt_ht,d_output_vocab_indices_01_single,d_d_ERRt_ht);

	if(lower_layer.copy_d_Err_ht) {
		cudaMemcpyAsync(lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht, d_d_ERRt_ht, Embedding_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s_layer_info.s1);
	}
	else {
		lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht = d_d_ERRt_ht;
	}
	
	cudaEventRecord(s_layer_info.d_ERR_ht_done,s_layer_info.s1);
}

template<typename dType>
void softmax_layer<dType>::compute_D_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D_grad,dType *d_h_t) {

	cudaSetDevice(s_layer_info.device_number);
	//zero out h_t
	cudaStreamWaitEvent(s_layer_info.s2,s_layer_info.outputdist_done,0);
	int threads_per_block = 128;
	int num_block = (Embedding_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	zero_columns_kernel_128<<<kernel_dim,threads_per_block,0,s_layer_info.s2 >>>(Embedding_size,d_h_t,d_output_vocab_indices_01_single,d_h_t);
	CUDA_GET_LAST_ERROR("zero out h_t");

	//multiply output dist and h_t
	dType alpha = -1;
	dType beta = 1;
	//d_D_grad = -1 * d_outputdist * d_h_t.T + 1 * d_D_grad
	cublasSetStream(s_layer_info.handle,s_layer_info.s2);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,output_vocab_size,Embedding_size,minibatch_size,&alpha,d_outputdist,output_vocab_size,
		d_h_t,Embedding_size,&beta,d_D_grad,output_vocab_size),"computing softmax D gradient failed in cuBLAS\n");

	//add columns of h_t to D_grad
	matrix_column_to_matrix_row_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s2 >>>(d_D_grad,d_h_t,d_D_grad,d_output_vocab_indices_single,Embedding_size,output_vocab_size);
	CUDA_GET_LAST_ERROR("add columns of h_t to D_grad");

	cudaEventRecord(s_layer_info.d_D_grad_done,s_layer_info.s2);
	
	cudaSetDevice(0);
}

template<typename dType>
void softmax_layer<dType>::compute_b_d_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_b_d_grad) {

	cudaSetDevice(s_layer_info.device_number);

	cudaStreamWaitEvent(s_layer_info.s3,s_layer_info.outputdist_done,0);

	//multiply
	dType alpha = -1;
	dType beta = 1;
	cublasSetStream(s_layer_info.handle,s_layer_info.s3);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(s_layer_info.handle,CUBLAS_OP_N,output_vocab_size,minibatch_size,&alpha,d_outputdist,output_vocab_size,
		d_output_vocab_indices_01_float_single,1,&beta,d_b_d_grad,1),"cuBLAS compute b_d_gradient failed");

	//add ones
	int threads_per_block = 128;
	int num_block = (minibatch_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(1,num_block,1);
	add_ones_b_d_grad<<< kernel_dim,threads_per_block,0,s_layer_info.s3>>>(d_b_d_grad,d_output_vocab_indices_01_single,d_output_vocab_indices_single,minibatch_size);

	cudaEventRecord(s_layer_info.d_b_d_grad_done,s_layer_info.s3);
}


template<typename dType>
cudaEvent_t softmax_layer<dType>::get_ERR_ht_event() {
	return s_layer_info.d_ERR_ht_done;
}


template<typename dType>
double softmax_layer<dType>::get_train_perplexity() {
	cudaSetDevice(s_layer_info.device_number);
	double tmp_perp;
	cudaMemcpy(&tmp_perp,d_train_perplexity,1*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemset(d_train_perplexity,0,1*sizeof(double));
	return tmp_perp;
}


template<typename dType>
void softmax_layer<dType>::clear_gradients() {
	clear_gradients_GPU();
}

template<typename dType>
void softmax_layer<dType>::clear_gradients_GPU() {

	cudaSetDevice(s_layer_info.device_number);

	cudaMemsetAsync(d_D_grad,0,output_vocab_size*Embedding_size*sizeof(dType),s_layer_info.s0);
	cudaMemsetAsync(d_b_d_grad,0,output_vocab_size*1*sizeof(dType),s_layer_info.s1);
	
	cudaDeviceSynchronize();
}


template<typename dType>
void softmax_layer<dType>::update_weights(int time_inex) {
	update_weights_GPU(time_inex);
}

template<typename dType>
void softmax_layer<dType>::update_weights_GPU(int time_inex) {

	cudaSetDevice(s_layer_info.device_number);
	
	// == scale_gradients()
	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_D_grad,thrust_d_D_grad + output_vocab_size*Embedding_size,unary_op);
	thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + output_vocab_size*1,unary_op);

	if(clip_gradients) {
		norm_clip_GPU_v2(thrust_d_D_grad,d_D_grad,norm_clip,output_vocab_size*Embedding_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);
	
	}

	if(ADAM) {

		//for update
		//alpha_adam = learning_rate;

		int threads_per_block = 128;
		int num_block = (output_vocab_size+threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block, threads_per_block);
	
		update_params_adam<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D, d_D_grad, d_D_mt, d_D_vt, alpha_adam, beta_1, beta_2, epsilon, time_inex, output_vocab_size, Embedding_size);	
	
		update_params_adam<<<kernel,threads_per_block,0,s_layer_info.s1>>>(d_b_d, d_b_d_grad, d_b_d_mt, d_b_d_vt, alpha_adam, beta_1, beta_2, epsilon, time_inex, output_vocab_size, 1);	
	
	}
	else { //SGD
		
		dType alpha = learning_rate;
		dType beta = 1;
		
		// == update_params()
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		//d_D = alpha * d_D_grad + d_D
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, Embedding_size, &alpha, 
			d_D_grad, output_vocab_size, &beta, d_D, output_vocab_size, d_D, output_vocab_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(s_layer_info.handle,s_layer_info.s1);
		//d_b_d = alpha * d_b_d_grad + d_b_d
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, 1, &alpha, d_b_d_grad, output_vocab_size, &beta, 
			d_b_d, output_vocab_size, d_b_d, output_vocab_size),"CUBLAS addition update parameter failed\n");
	}
}

template<typename dType>
void softmax_layer<dType>::update_learning_rate(dType learning_rate) {
	this->learning_rate = learning_rate;
}

template<typename dType>
void softmax_layer<dType>::adam_switch_sgd() {
	this->ADAM = false;
}


template<typename dType>
void softmax_layer<dType>::dump_weights_GPU(ofstream &output) {
	cudaSetDevice(s_layer_info.device_number);

	write_matrix_GPU(d_D,output_vocab_size,Embedding_size,output);
	write_matrix_GPU(d_b_d,output_vocab_size,1,output);
}

template<typename dType>
void softmax_layer<dType>::dump_weights(ofstream &output) {

	dump_weights_GPU(output);
}


template<typename dType>
void softmax_layer<dType>::load_weights_GPU(ifstream &input) {

	cudaSetDevice(s_layer_info.device_number);
	
	read_matrix_GPU(d_D,output_vocab_size,Embedding_size,input);
	//thrust::device_ptr<dType> temp_ptr = thrust::device_pointer_cast(d_D);
	read_matrix_GPU(d_b_d,output_vocab_size,1,input);

}

template<typename dType>
void softmax_layer<dType>::load_weights(ifstream &input) {

	load_weights_GPU(input);
}


template<typename dType>
void softmax_layer<dType>::get_perplexity_GPU(dType *d_h_t,int index) {	

	devSynchAll();
	
	//cout<<"show d_output_vocab_indices_single get0: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);

	//multiply the D matrix with the hidden state matrix
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	
	//cout<<"show d_h_t: "<<endl;
	//show_matrix(d_h_t, Embedding_size, minibatch_size);
	
	//cout<<"show d_D: "<<endl;
	//show_matrix(d_D, output_vocab_size, Embedding_size);
	
	//d_outputdist = d_D *d_h_t
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,
	 output_vocab_size, minibatch_size, Embedding_size, &alpha, d_D, output_vocab_size,
	  d_h_t,Embedding_size, &beta, d_outputdist, output_vocab_size),"get_distribution cuBLAS call failed 2\n");

	//cout<<"test 1 get_perplexity_GPU"<<endl;
	
	//cout<<"show d_outputdist: "<<endl;
	//show_matrix_double(d_outputdist, output_vocab_size, minibatch_size);
	
	//cout<<"show d_output_vocab_indices_single get1: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);
	
	//add the bias vector to the matrix
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_bias_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s0 >>>(output_vocab_size,d_outputdist,d_b_d,d_outputdist);
	CUDA_GET_LAST_ERROR("perplexity bias");
	
	//cout<<"show d_output_vocab_indices_single get2: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);

	
	outputdist_perplexity_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist_perp, d_outputdist, output_vocab_size,false,NULL);
	CUDA_GET_LAST_ERROR("Perplexity Kernel");
	
	//cout<<"d_outputdist_perp: "<<endl;
	//show_matrix_double(d_outputdist_perp, output_vocab_size, minibatch_size);
	
	//cout<<"show d_output_vocab_indices_single get3: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);

	cudaDeviceSynchronize();
}

template<typename dType>
double softmax_layer<dType>::compute_loss_GPU(int index) {
	
	cudaSetDevice(s_layer_info.device_number);
	
	double loss = 0;
	devSynchAll();
	
	//cout<<"show d_output_vocab_indices_single pre pre: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);

	get_perplexity_GPU(nodes[index].d_h_t,index);
	
	//cout<<"test 1 compute_loss_GPU"<<endl;
	
	//cout<<"show d_output_vocab_indices_single pre: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);

	cudaSetDevice(s_layer_info.device_number);
	devSynchAll();
	thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_output_vocab_indices_single);
	thrust::device_ptr<int> d_ptr_01 = thrust::device_pointer_cast(d_output_vocab_indices_01_single);
	
	thrust::device_ptr<double> d_ptr_sm = thrust::device_pointer_cast(d_outputdist_perp);
	
	//cout<<"test 2 compute_loss_GPU"<<endl;
	
	//cout<<"show d_output_vocab_indices_single: "<<endl;
	//show_matrix_int(d_output_vocab_indices_single, minibatch_size, 10);
	
	//cout<<"show d_output_vocab_indices_01_single: "<<endl;
	//show_matrix_int(d_output_vocab_indices_01_single, minibatch_size, 1);

	//cout<<"show d_outputdist_perp: "<<endl;
	//show_matrix_double(d_outputdist_perp,output_vocab_size, minibatch_size);
	
	//cout<<"test 2.01 compute_loss_GPU: "<<d_ptr[2]<<endl;
	//cout<<"test 2.02 compute_loss_GPU: "<<d_ptr_01[0]<<endl;
	//cout<<"test 2.03 compute_loss_GPU: "<<d_ptr_sm[0]<<endl;
	
	for(int i=0; i < minibatch_size; i++) {
		
		//cout<<"test 2.1 pre"<<endl;
		//cout<<"test 2.1 compute_loss_GPU: "<<d_ptr_01[i]<<endl;
		
		if(d_ptr_01[i]==1) {
			//loss+=std::log((double)d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)]);
			
			//cout<<"test 3 compute_loss_GPU "<<i<<endl;
			//cout<<"test 3.1 compute_loss_GPU "<<i<<": "<<d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)]<<endl;
			loss += d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)];
			
			//cout<<"test 4 compute_loss_GPU "<<i<<endl;
			
		}
	}
	
	return loss;

}

template<typename dType>
void softmax_layer<dType>::check_all_gradients(dType epsilon) 
{	
	check_all_gradients_GPU(epsilon);
}

template<typename dType>
void softmax_layer<dType>::check_all_gradients_GPU(dType epsilon) 
{	
	cudaSetDevice(s_layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR SOFTMAX LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR D\n";
	check_gradient_GPU(epsilon,d_D,d_D_grad,output_vocab_size,Embedding_size);
	cudaSetDevice(s_layer_info.device_number);
		
	std::cout << "GRADIENT CHECKING FOR b_d\n";
	check_gradient_GPU(epsilon,d_b_d,d_b_d_grad,output_vocab_size,1);
	cudaSetDevice(s_layer_info.device_number);

}

template<typename dType>
void softmax_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	cudaSetDevice(s_layer_info.device_number);
	cudaDeviceSynchronize();
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<min(5,rows); i++) {
		for(int j=0; j<min(2,cols); j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError();
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError();
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}




