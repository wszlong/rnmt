

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::init_Hidden_To_Hidden_Layer(int Embedding_size,int LSTM_size,int minibatch_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed,bool dropout,dType dropout_rate,int layer_number,global_params &params)
{

	//Set the debug mode
	debug = debug_temp;
	this->minibatch_size = minibatch_size;
	this->learning_rate = learning_rate;
	this->clip_gradients = clip_gradients;
	this->norm_clip = norm_clip;
	this->model = model;
	this->Embedding_size = Embedding_size;
	this->LSTM_size = LSTM_size;
	this->longest_sent = longest_sent;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;
	this->layer_number = layer_number;
	//adam
	this->ADAM = params.ADAM;
	this->alpha_adam = params.alpha_adam;
	this->beta_1 = params.beta_1;
	this->beta_2 = params.beta_2;
	this->epsilon = params.epsilon;
	gen.seed(seed);

	init_Hidden_To_Hidden_Layer_GPU(Embedding_size,LSTM_size,minibatch_size,
 		longest_sent,debug_temp,learning_rate,clip_gradients,norm_clip,
 		model,seed);

	//Initialize the vector of LSTM nodes to longest sentence
	nodes.clear();
	for(int i=0;i < longest_sent; i++) {
		nodes.push_back(LSTM_HH_Node<dType>(Embedding_size,LSTM_size,minibatch_size,this,i,dropout,dropout_rate));
	}
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::init_Hidden_To_Hidden_Layer_GPU(int Embedding_size,int LSTM_size,int minibatch_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed)
{
	cudaSetDevice(hh_layer_info.device_number);

	full_matrix_setup(&h_W_ho,&d_W_ho,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf,&d_W_hf,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi,&d_W_hi,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc,&d_W_hc,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi_grad,&d_W_hi_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf_grad,&d_W_hf_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc_grad,&d_W_hc_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_ho_grad,&d_W_ho_grad,LSTM_size,LSTM_size);

	full_matrix_setup(&h_M_i,&d_M_i,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_f,&d_M_f,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_o,&d_M_o,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_c,&d_M_c,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_i_grad,&d_M_i_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_f_grad,&d_M_f_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_o_grad,&d_M_o_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_c_grad,&d_M_c_grad,LSTM_size,LSTM_size);
	
	//adam
	if(ADAM) {
		full_matrix_setup_0(&h_W_hi,&d_W_hi_mt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_hi,&d_W_hi_vt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_hf,&d_W_hf_mt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_hf,&d_W_hf_vt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_hc,&d_W_hc_mt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_hc,&d_W_hc_vt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_ho,&d_W_ho_mt,LSTM_size,LSTM_size);
		full_matrix_setup_0(&h_W_ho,&d_W_ho_vt,LSTM_size,LSTM_size);
		
		full_matrix_setup_0(&h_M_i,&d_M_i_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_i,&d_M_i_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_f,&d_M_f_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_f,&d_M_f_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_o,&d_M_o_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_o,&d_M_o_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_c,&d_M_c_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_M_c,&d_M_c_vt,LSTM_size,Embedding_size);
		
		full_matrix_setup_0(&h_b_i,&d_b_i_mt,LSTM_size,1);
		full_matrix_setup_0(&h_b_i,&d_b_i_vt,LSTM_size,1);
		full_matrix_setup_0(&h_b_f,&d_b_f_mt,LSTM_size,1);
		full_matrix_setup_0(&h_b_f,&d_b_f_vt,LSTM_size,1);
		full_matrix_setup_0(&h_b_c,&d_b_c_mt,LSTM_size,1);
		full_matrix_setup_0(&h_b_c,&d_b_c_vt,LSTM_size,1);
		full_matrix_setup_0(&h_b_o,&d_b_o_mt,LSTM_size,1);
		full_matrix_setup_0(&h_b_o,&d_b_o_vt,LSTM_size,1);

		//bidirection
		if(lower_layer.lower_input && upper_layer.source_side){

			full_matrix_setup_0(&h_M_i,&d_U_i_mt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_i,&d_U_i_vt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_f,&d_U_f_mt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_f,&d_U_f_vt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_o,&d_U_o_mt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_o,&d_U_o_vt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_c,&d_U_c_mt,LSTM_size,Embedding_size);
			full_matrix_setup_0(&h_M_c,&d_U_c_vt,LSTM_size,Embedding_size);
		}
	}
	
	//bidirection
	if(lower_layer.lower_input && upper_layer.source_side){
		full_matrix_setup(&h_M_i,&d_U_i,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_f,&d_U_f,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_o,&d_U_o,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_c,&d_U_c,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_i_grad,&d_U_i_grad,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_f_grad,&d_U_f_grad,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_o_grad,&d_U_o_grad,LSTM_size,LSTM_size);
		full_matrix_setup(&h_M_c_grad,&d_U_c_grad,LSTM_size,LSTM_size);
	}

	full_matrix_setup(&h_b_i,&d_b_i,LSTM_size,1);
	full_matrix_setup(&h_b_f,&d_b_f,LSTM_size,1);
	full_matrix_setup(&h_b_c,&d_b_c,LSTM_size,1);
	full_matrix_setup(&h_b_o,&d_b_o,LSTM_size,1);
	full_matrix_setup(&h_b_i_grad,&d_b_i_grad,LSTM_size,1);
	full_matrix_setup(&h_b_f_grad,&d_b_f_grad,LSTM_size,1);
	full_matrix_setup(&h_b_c_grad,&d_b_c_grad,LSTM_size,1);
	full_matrix_setup(&h_b_o_grad,&d_b_o_grad,LSTM_size,1);

	full_matrix_setup_0(&h_init_hidden_vector,&d_init_hidden_vector,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_cell_vector,&d_init_cell_vector,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_d_ERRnTOtp1_ht,&d_init_d_ERRnTOtp1_ht,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_d_ERRnTOtp1_ct,&d_init_d_ERRnTOtp1_ct,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp1,&d_temp1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp2,&d_temp2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp3,&d_temp3,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp4,&d_temp4,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp5,&d_temp5,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp6,&d_temp6,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp7,&d_temp7,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp8,&d_temp8,LSTM_size,minibatch_size);
	
	//bidirection
	if(lower_layer.lower_input && upper_layer.source_side){
		full_matrix_setup(&h_temp1,&d_temp1_bi,LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp3,&d_temp3_bi,LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp5,&d_temp5_bi,LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp7,&d_temp7_bi,LSTM_size,minibatch_size);
	}

	full_matrix_setup(&h_h_t_below,&d_h_t_below,LSTM_size,minibatch_size);

	full_matrix_setup_0(&h_input_vocab_indicies,&d_input_vocab_indicies,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indices_01_full,&d_input_vocab_indices_01_full,minibatch_size,longest_sent);

	//set to all ones
	full_vector_setup_ones(&h_ones_minibatch,&d_ones_minibatch,minibatch_size);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
	
	
	//get device pointers
	thrust_d_W_ho_grad = thrust::device_pointer_cast(d_W_ho_grad); 
	thrust_d_W_hf_grad = thrust::device_pointer_cast(d_W_hf_grad);
	thrust_d_W_hi_grad = thrust::device_pointer_cast(d_W_hi_grad); 
	thrust_d_W_hc_grad = thrust::device_pointer_cast(d_W_hc_grad);

	thrust_d_M_i_grad = thrust::device_pointer_cast(d_M_i_grad);
	thrust_d_M_f_grad = thrust::device_pointer_cast(d_M_f_grad);
	thrust_d_M_o_grad = thrust::device_pointer_cast(d_M_o_grad);
	thrust_d_M_c_grad = thrust::device_pointer_cast(d_M_c_grad);
	
	//bidirection
	if(lower_layer.lower_input && upper_layer.source_side){
		thrust_d_U_i_grad = thrust::device_pointer_cast(d_U_i_grad);
		thrust_d_U_f_grad = thrust::device_pointer_cast(d_U_f_grad);
		thrust_d_U_o_grad = thrust::device_pointer_cast(d_U_o_grad);
		thrust_d_U_c_grad = thrust::device_pointer_cast(d_U_c_grad);
	}

	thrust_d_b_i_grad = thrust::device_pointer_cast(d_b_i_grad);
	thrust_d_b_f_grad = thrust::device_pointer_cast(d_b_f_grad);
	thrust_d_b_c_grad = thrust::device_pointer_cast(d_b_c_grad);
	thrust_d_b_o_grad = thrust::device_pointer_cast(d_b_o_grad);
	
	//for saving space in the LSTM
	full_matrix_setup(&h_d_ERRnTOt_ht,&d_d_ERRnTOt_ht,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRt_ct,&d_d_ERRt_ct,LSTM_size,minibatch_size);
	
	full_matrix_setup(&h_d_ERRnTOt_ct,&d_d_ERRnTOt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ot,&d_d_ERRnTOt_ot,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ft,&d_d_ERRnTOt_ft,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_tanhcpt,&d_d_ERRnTOt_tanhcpt,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_it,&d_d_ERRnTOt_it,LSTM_size,minibatch_size);
	
	full_matrix_setup(&h_d_ERRnTOt_htM1,&d_d_ERRnTOt_htM1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ctM1,&d_d_ERRnTOt_ctM1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_h_Below,&d_d_ERRnTOt_h_Below,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_h_Below,&d_d_ERRnTOt_h_Below_bi,LSTM_size,minibatch_size); //bi


	curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);
	boost::uniform_int<> unif_boost( 1, 1000000 );
	curandSetPseudoRandomGeneratorSeed(rand_gen,MY_CUDA::curr_seed);
	MY_CUDA::curr_seed+=7;

	clear_gradients(true);

	cudaSetDevice(hh_layer_info.device_number);
	cudaDeviceSynchronize();

}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::init_attention(int device_number,bool feed_input,neuralMT_model<dType> *model,global_params &params) {

	attent_layer = new attention_layer<dType>(Embedding_size,LSTM_size,minibatch_size,hh_layer_info.device_number,longest_sent,hh_layer_info.handle,model,
		feed_input,clip_gradients,norm_clip,dropout,dropout_rate,params);


	//now switch on the attention flag in the attention nodes
	for(int i=0; i<nodes.size(); i++) {
		nodes[i].attention_model = true;
	}
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::prep_GPU_vocab_indices(int *h_input_vocab_indicies,int current_length) {

	cudaSetDevice(hh_layer_info.device_number);

	this->h_input_vocab_indicies = h_input_vocab_indicies;
	this->current_length = current_length;

	//transfer to the GPU
	cudaMemcpy(d_input_vocab_indicies, h_input_vocab_indicies, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer");

	//Launch kernel to turn into 0/1's and indicies with no -1's
	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid = 128;
	vocab_to_01<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_01_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 1");


	if(attent_layer!=NULL) {
		attent_layer->transfer_done = false; //!
	}

}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::clear_gradients(bool init) {
	clear_gradients_GPU(init);
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::clear_gradients_GPU(bool init) {

	cudaSetDevice(hh_layer_info.device_number);

	cudaDeviceSynchronize();
	cudaMemsetAsync(d_W_hi_grad, 0, LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s0);
	cudaMemsetAsync(d_b_i_grad, 0, LSTM_size*1*sizeof(dType),hh_layer_info.s1);

	cudaMemsetAsync(d_W_hf_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s2);
	cudaMemsetAsync(d_b_f_grad,0,LSTM_size*1*sizeof(dType),hh_layer_info.s3);

	cudaMemsetAsync(d_W_hc_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s4);
	cudaMemsetAsync(d_b_c_grad,0,LSTM_size*1*sizeof(dType),hh_layer_info.s5);

	cudaMemsetAsync(d_W_ho_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s6);
	cudaMemsetAsync(d_b_o_grad,0,LSTM_size*1*sizeof(dType),hh_layer_info.s7);

	cudaMemsetAsync(d_M_i_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s9);
	cudaMemsetAsync(d_M_f_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s10);
	cudaMemsetAsync(d_M_o_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s11);
	cudaMemsetAsync(d_M_c_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s12);
	
	//bidirection
	if(lower_layer.lower_input && upper_layer.source_side){
		cudaMemsetAsync(d_U_i_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s9);
		cudaMemsetAsync(d_U_f_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s10);
		cudaMemsetAsync(d_U_o_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s11);
		cudaMemsetAsync(d_U_c_grad,0,LSTM_size*LSTM_size*sizeof(dType),hh_layer_info.s12);
	}

	if(attent_layer!=NULL) {
		attent_layer->clear_gradients();
	}

	devSynchAll();
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::update_params(int time_index) {

	dType alpha = learning_rate;
	dType beta = 1;

	devSynchAll();

	if(ADAM) {
		
		//for update
		//alpha_adam  = learning_rate;

		int threads_per_block = 128;
		int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block, threads_per_block);
		
		//d_W_
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s0>>>(d_W_hi, d_W_hi_grad, d_W_hi_mt, d_W_hi_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s2>>>(d_W_hf, d_W_hf_grad, d_W_hf_mt, d_W_hf_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s4>>>(d_W_hc, d_W_hc_grad, d_W_hc_mt, d_W_hc_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s6>>>(d_W_ho, d_W_ho_grad, d_W_ho_mt, d_W_ho_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		//d_M
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s9>>>(d_M_i, d_M_i_grad, d_M_i_mt, d_M_i_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s10>>>(d_M_f, d_M_f_grad, d_M_f_mt, d_M_f_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s12>>>(d_M_o, d_M_o_grad, d_M_o_mt, d_M_o_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s11>>>(d_M_c, d_M_c_grad, d_M_c_mt, d_M_c_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		

		//bidirection
		if(lower_layer.lower_input && upper_layer.source_side){
		
			update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s9>>>(d_U_i, d_U_i_grad, d_U_i_mt, d_U_i_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
			
			update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s10>>>(d_U_f, d_U_f_grad, d_U_f_mt, d_U_f_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
			
			update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s12>>>(d_U_o, d_U_o_grad, d_U_o_mt, d_U_o_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
			
			update_params_adam<<<kernel,threads_per_block,0,hh_layer_info.s11>>>(d_U_c, d_U_c_grad, d_U_c_mt, d_U_c_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		}
	
		//d_b	
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s1>>>(d_b_i,d_b_i_grad,d_b_i_mt,d_b_i_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_i");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s3>>>(d_b_f,d_b_f_grad,d_b_f_mt,d_b_f_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_f");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s5>>>(d_b_c,d_b_c_grad,d_b_c_mt,d_b_c_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_c");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s7>>>(d_b_o,d_b_o_grad,d_b_o_mt,d_b_o_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_o");
			

	}
	else { //SGD
	
		//normal matrices
		cublasSetStream(hh_layer_info.handle,hh_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hi_grad, LSTM_size, &beta, d_W_hi, LSTM_size, d_W_hi, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s2);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hf_grad, LSTM_size, &beta, d_W_hf, LSTM_size, d_W_hf, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s4);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hc_grad, LSTM_size, &beta, d_W_hc, LSTM_size, d_W_hc, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s6);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_ho_grad, LSTM_size, &beta, d_W_ho, LSTM_size, d_W_ho, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s9);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_M_i_grad, LSTM_size, &beta, d_M_i, LSTM_size, d_M_i, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s10);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_M_f_grad, LSTM_size, &beta, d_M_f, LSTM_size, d_M_f, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s12);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_M_c_grad, LSTM_size, &beta, d_M_c, LSTM_size, d_M_c, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(hh_layer_info.handle,hh_layer_info.s11);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_M_o_grad, LSTM_size, &beta, d_M_o, LSTM_size, d_M_o, LSTM_size),"CUBLAS addition update parameter failed\n");

		//bidirection
		if(lower_layer.lower_input && upper_layer.source_side){
			cublasSetStream(hh_layer_info.handle,hh_layer_info.s9);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
				d_U_i_grad, LSTM_size, &beta, d_U_i, LSTM_size, d_U_i, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(hh_layer_info.handle,hh_layer_info.s10);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
				d_U_f_grad, LSTM_size, &beta, d_U_f, LSTM_size, d_U_f, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(hh_layer_info.handle,hh_layer_info.s12);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
				d_U_c_grad, LSTM_size, &beta, d_U_c, LSTM_size, d_U_c, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(hh_layer_info.handle,hh_layer_info.s11);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(hh_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
				d_U_o_grad, LSTM_size, &beta, d_U_o, LSTM_size, d_U_o, LSTM_size),"CUBLAS addition update parameter failed\n");	
		}

		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s1>>>(d_b_i,d_b_i_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("H_H update d_b_i");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s3>>>(d_b_f,d_b_f_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("H_H update d_b_f");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s5>>>(d_b_c,d_b_c_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("H_H update d_b_c");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,hh_layer_info.s7>>>(d_b_o,d_b_o_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("H_H update d_b_o");
		
	}
	
	if(attent_layer!=NULL) {
		attent_layer->update_params(time_index);
	}
	
	devSynchAll();
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::scale_gradients() {
	
	cudaSetDevice(hh_layer_info.device_number);

	scale_functor unary_op(minibatch_size);

	thrust::for_each(thrust_d_W_hi_grad,thrust_d_W_hi_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_i_grad,thrust_d_b_i_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hf_grad,thrust_d_W_hf_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_f_grad,thrust_d_b_f_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hc_grad,thrust_d_W_hc_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_c_grad,thrust_d_b_c_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_ho_grad,thrust_d_W_ho_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_o_grad,thrust_d_b_o_grad + LSTM_size*1,unary_op);


	thrust::for_each(thrust_d_M_i_grad,thrust_d_M_i_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_grad,thrust_d_M_f_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_grad,thrust_d_M_o_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_grad,thrust_d_M_c_grad + LSTM_size*LSTM_size,unary_op);
	
	if(lower_layer.lower_input && upper_layer.source_side){
		thrust::for_each(thrust_d_U_i_grad,thrust_d_U_i_grad + LSTM_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_U_f_grad,thrust_d_U_f_grad + LSTM_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_U_o_grad,thrust_d_U_o_grad + LSTM_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_U_c_grad,thrust_d_U_c_grad + LSTM_size*LSTM_size,unary_op);
	}

	if(attent_layer!=NULL) {
		attent_layer->scale_gradients();
	}

	devSynchAll();
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::update_weights(int time_index) {
	update_weights_GPU(time_index);
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::update_weights_GPU(int time_index) {

	cudaSetDevice(hh_layer_info.device_number);

	scale_gradients();

	if(clip_gradients) {
		
		norm_clip_GPU_v2(thrust_d_W_hi_grad,d_W_hi_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_hf_grad,d_W_hf_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_hc_grad,d_W_hc_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_ho_grad,d_W_ho_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

		norm_clip_GPU_v2(thrust_d_b_i_grad,d_b_i_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_f_grad,d_b_f_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_c_grad,d_b_c_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_o_grad,d_b_o_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);

		norm_clip_GPU_v2(thrust_d_M_i_grad,d_M_i_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_f_grad,d_M_f_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_o_grad,d_M_o_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_c_grad,d_M_c_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	
		if(lower_layer.lower_input && upper_layer.source_side){
			norm_clip_GPU_v2(thrust_d_U_i_grad,d_U_i_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_U_f_grad,d_U_f_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_U_o_grad,d_U_o_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_U_c_grad,d_U_c_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		}

		if(attent_layer!=NULL) {
			attent_layer->clip_gradients_func();
		}	
	}

	update_params(time_index); //!
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::zero_attent_error() {
	cudaSetDevice(hh_layer_info.device_number);
	for(int i=0; i<nodes.size(); i++) {
		cudaMemset(nodes[i].d_d_ERRt_ht,0,LSTM_size*minibatch_size*sizeof(dType));
	}
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::dump_weights_GPU(ofstream &output) {

	cudaSetDevice(hh_layer_info.device_number);

	write_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_i,LSTM_size,1,output);

	write_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_f,LSTM_size,1,output);

	write_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_c,LSTM_size,1,output);

	write_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_o,LSTM_size,1,output);

	write_matrix_GPU(d_M_i,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_f,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_o,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_c,LSTM_size,LSTM_size,output);
	
	if(lower_layer.lower_input && upper_layer.source_side){
		write_matrix_GPU(d_U_i,LSTM_size,LSTM_size,output);
		write_matrix_GPU(d_U_f,LSTM_size,LSTM_size,output);
		write_matrix_GPU(d_U_o,LSTM_size,LSTM_size,output);
		write_matrix_GPU(d_U_c,LSTM_size,LSTM_size,output);
	}

	if(attent_layer!=NULL) {
		attent_layer->dump_weights(output);
	}
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::dump_weights(ofstream &output) {
	dump_weights_GPU(output);
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::load_weights_GPU(ifstream &input) {

	cudaSetDevice(hh_layer_info.device_number);

	read_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_i,LSTM_size,1,input);

	read_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);

	read_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);

	read_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);

	read_matrix_GPU(d_M_i,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_f,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_o,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_c,LSTM_size,LSTM_size,input);
	
	if(lower_layer.lower_input && upper_layer.source_side){
		read_matrix_GPU(d_U_i,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_f,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_o,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_c,LSTM_size,LSTM_size,input);
	}

	if(attent_layer!=NULL) {
		attent_layer->load_weights(input);
	}
}

template<typename dType>
void Hidden_To_Hidden_Layer<dType>::load_weights(std::ifstream &input) {
	load_weights_GPU(input);
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::check_all_gradients(dType epsilon) {
	check_all_gradients_GPU(epsilon);
}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::check_all_gradients_GPU(dType epsilon) {

	cudaSetDevice(hh_layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR HIDDEN LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR W_hi\n";
	check_gradient_GPU(epsilon,d_W_hi,d_W_hi_grad,LSTM_size,LSTM_size);
	/*
	std::cout << "GRADIENT CHECKING FOR W_hf\n";
	check_gradient_GPU(epsilon,d_W_hf,d_W_hf_grad,LSTM_size,LSTM_size);

	std::cout << "GRADIENT CHECKING FOR W_ho\n";
	check_gradient_GPU(epsilon,d_W_ho,d_W_ho_grad,LSTM_size,LSTM_size);

	std::cout << "GRADIENT CHECKING FOR W_hc\n";
	check_gradient_GPU(epsilon,d_W_hc,d_W_hc_grad,LSTM_size,LSTM_size);

	std::cout << "GRADIENT CHECKING FOR b_i\n";
	check_gradient_GPU(epsilon,d_b_i,d_b_i_grad,LSTM_size,1);

	std::cout << "GRADIENT CHECKING FOR b_f\n";
	check_gradient_GPU(epsilon,d_b_f,d_b_f_grad,LSTM_size,1);

	std::cout << "GRADIENT CHECKING FOR b_c\n";
	check_gradient_GPU(epsilon,d_b_c,d_b_c_grad,LSTM_size,1);

	std::cout << "GRADIENT CHECKING FOR b_o\n";
	check_gradient_GPU(epsilon,d_b_o,d_b_o_grad,LSTM_size,1);

	std::cout << "GRADIENT CHECKING FOR M_i\n";
	check_gradient_GPU(epsilon,d_M_i,d_M_i_grad,LSTM_size,LSTM_size);
	
	std::cout << "GRADIENT CHECKING FOR M_f\n";
	check_gradient_GPU(epsilon,d_M_f,d_M_f_grad,LSTM_size,LSTM_size);

	std::cout << "GRADIENT CHECKING FOR M_o\n";
	check_gradient_GPU(epsilon,d_M_o,d_M_o_grad,LSTM_size,LSTM_size);
	
	std::cout << "GRADIENT CHECKING FOR M_c\n";
	check_gradient_GPU(epsilon,d_M_c,d_M_c_grad,LSTM_size,LSTM_size);
	*/

	if(lower_layer.lower_input && upper_layer.source_side) {	
		std::cout << "GRADIENT CHECKING FOR U_i\n";
		check_gradient_GPU(epsilon,d_U_i,d_U_i_grad,LSTM_size,LSTM_size);
		
		//std::cout << "GRADIENT CHECKING FOR U_f\n";
		//check_gradient_GPU(epsilon,d_U_f,d_U_f_grad,LSTM_size,LSTM_size);

		//std::cout << "GRADIENT CHECKING FOR U_o\n";
		//check_gradient_GPU(epsilon,d_U_o,d_U_o_grad,LSTM_size,LSTM_size);
		
		//std::cout << "GRADIENT CHECKING FOR U_c\n";
		//check_gradient_GPU(epsilon,d_U_c,d_U_c_grad,LSTM_size,LSTM_size);
	}

	if(attent_layer!=NULL) {
		attent_layer->check_gradients(epsilon);
	}

}


template<typename dType>
void Hidden_To_Hidden_Layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	cudaSetDevice(hh_layer_info.device_number);
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<min(5,rows); i++) {
		for(int j=0; j<min(2,cols); j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError();
			cudaSetDevice(hh_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError();
			cudaSetDevice(hh_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			//std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] <<  "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
			else if(d_thrust_grad[IDX2C(i,j,rows)]==0 ||loss/(2*epsilon) ==0) {
				std::cout << "ZERO GRADIENTS\n";
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}







