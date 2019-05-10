
template<typename dType>
void Input_To_Hidden_Layer<dType>::init_Input_To_Hidden_Layer(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed,bool dropout,dType dropout_rate,bool bi_dir,global_params &params,bool source)
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
	this->bi_dir = bi_dir;
	//adam
	this->ADAM = params.ADAM;
	this->alpha_adam = params.alpha_adam;
	this->beta_1 = params.beta_1;
	this->beta_2 = params.beta_2;
	this->epsilon = params.epsilon;
	gen.seed(seed);

	init_Input_To_Hidden_Layer_GPU(Embedding_size,LSTM_size,minibatch_size,vocab_size,
 		longest_sent,debug_temp,learning_rate,clip_gradients,norm_clip,
 		model,seed,params,source);

	//Initialize the vector of LSTM nodes to longest sentence
	nodes.clear();
	for(int i=0;i < longest_sent; i++) {
		nodes.push_back(LSTM_IH_Node<dType>(Embedding_size,LSTM_size,minibatch_size,vocab_size,this,i,dropout,dropout_rate,bi_dir));
	}
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::init_Input_To_Hidden_Layer_GPU(int Embedding_size,int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed,global_params &params,bool source)
{

	cudaSetDevice(ih_layer_info.device_number);

	full_matrix_setup(&h_W_ho,&d_W_ho,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf,&d_W_hf,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi,&d_W_hi,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc,&d_W_hc,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi_grad,&d_W_hi_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf_grad,&d_W_hf_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc_grad,&d_W_hc_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_ho_grad,&d_W_ho_grad,LSTM_size,LSTM_size);

	full_matrix_setup(&h_M_i,&d_M_i,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_f,&d_M_f,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_o,&d_M_o,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_c,&d_M_c,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_i_grad,&d_M_i_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_f_grad,&d_M_f_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_o_grad,&d_M_o_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_M_c_grad,&d_M_c_grad,LSTM_size,Embedding_size);

	//cout<<"show d_M_i: "<<endl;
	//show_matrix(d_M_i, LSTM_size, Embedding_size);
		
	if(ADAM) {
		//Adam
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

		full_matrix_setup_0(&h_W,&d_W_mt,Embedding_size,vocab_size);
		full_matrix_setup_0(&h_W,&d_W_vt,Embedding_size,vocab_size);
	}

	full_matrix_setup(&h_b_i,&d_b_i,LSTM_size,1);
	full_matrix_setup(&h_b_f,&d_b_f,LSTM_size,1);
	full_matrix_setup(&h_b_c,&d_b_c,LSTM_size,1);
	full_matrix_setup(&h_b_o,&d_b_o,LSTM_size,1);
	full_matrix_setup(&h_b_i_grad,&d_b_i_grad,LSTM_size,1);
	full_matrix_setup(&h_b_f_grad,&d_b_f_grad,LSTM_size,1);
	full_matrix_setup(&h_b_c_grad,&d_b_c_grad,LSTM_size,1);
	full_matrix_setup(&h_b_o_grad,&d_b_o_grad,LSTM_size,1);
	
	if(bi_dir) {
		full_matrix_setup(&h_W,&d_W,Embedding_size,vocab_size);
		d_W = model->input_layer_source.d_W;
	}
	else {
		full_matrix_setup(&h_W,&d_W,Embedding_size,vocab_size);
	}

	input_vocab_size = vocab_size;

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

	full_matrix_setup_0(&h_input_vocab_indicies,&d_input_vocab_indicies,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indicies_bi,&d_input_vocab_indicies_bi,minibatch_size,longest_sent); //bidirection
	full_matrix_setup_0(&h_input_vocab_indices_full,&d_input_vocab_indices_full,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indices_01_full,&d_input_vocab_indices_01_full,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indicies_Wgrad,&d_input_vocab_indicies_Wgrad,minibatch_size,longest_sent);

	//set to all ones
	full_vector_setup_ones(&h_ones_minibatch,&d_ones_minibatch,minibatch_size);

	full_matrix_setup(&h_temp1,&d_small_W_grad,Embedding_size*minibatch_size,longest_sent);
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_reverse_unique_indicies, vocab_size*sizeof(int)),"GPU memory allocation failed\n");
	cudaMemset(d_small_W_grad,0,Embedding_size*longest_sent*minibatch_size*sizeof(dType));
	cudaMemset(d_reverse_unique_indicies,0,vocab_size*sizeof(int));
	
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
	
	thrust_d_small_W_grad = thrust::device_pointer_cast(d_small_W_grad);
	
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
	

	curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);
	boost::uniform_int<> unif_boost( 1, 1000000 );
	curandSetPseudoRandomGeneratorSeed(rand_gen,MY_CUDA::curr_seed);
	MY_CUDA::curr_seed+=7;

	clear_gradients(true);
	cudaSetDevice(ih_layer_info.device_number);
	cudaDeviceSynchronize();

}


//pass in the pointer pointing to h_tild in the loweest layer
template<typename dType>
void Input_To_Hidden_Layer<dType>::init_feed_input(Hidden_To_Hidden_Layer<dType> *hidden_layer) {
	
	this->feed_input = true;
	cout<<"Init feedinput"<<endl;		

	for(int i=0; i<nodes.size(); i++) {
		nodes[i].attention_extra(); //d_h_tild
	}
	
	for(int i=0; i<hidden_layer->nodes.size()-1; i++) {
		hidden_layer->attent_layer->nodes[i].feed_input_init(nodes[i+1].d_h_tild);
	}
	
	for(int i=0; i<hidden_layer->nodes.size()-1; i++) {
		nodes[i+1].d_ERRnTOt_h_tild_cpy = hidden_layer->attent_layer->nodes[i].d_ERRtTOn_htild_below;
	}


	cudaSetDevice(ih_layer_info.device_number);
	dType *h_temp;
	full_matrix_setup(&h_temp,&d_Q_i,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_f,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_o,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_c,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_i_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_f_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_o_grad,LSTM_size,Embedding_size);
	full_matrix_setup(&h_temp,&d_Q_c_grad,LSTM_size,Embedding_size);
	
	//adam
	if(ADAM) {
		full_matrix_setup_0(&h_temp,&d_Q_i_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_i_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_f_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_f_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_o_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_o_vt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_c_mt,LSTM_size,Embedding_size);
		full_matrix_setup_0(&h_temp,&d_Q_c_vt,LSTM_size,Embedding_size);
	}

	thrust_d_Q_i_grad = thrust::device_pointer_cast(d_Q_i_grad);
	thrust_d_Q_f_grad = thrust::device_pointer_cast(d_Q_f_grad);
	thrust_d_Q_o_grad = thrust::device_pointer_cast(d_Q_o_grad);
	thrust_d_Q_c_grad = thrust::device_pointer_cast(d_Q_c_grad);
	
	//
	cudaMemset(d_Q_i_grad,0,LSTM_size*Embedding_size*sizeof(dType));
	cudaMemset(d_Q_f_grad,0,LSTM_size*Embedding_size*sizeof(dType));
	cudaMemset(d_Q_o_grad,0,LSTM_size*Embedding_size*sizeof(dType));
	cudaMemset(d_Q_c_grad,0,LSTM_size*Embedding_size*sizeof(dType));
	
	full_matrix_setup(&h_temp,&d_temp9,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp10,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp11,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp12,LSTM_size,minibatch_size);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::prep_GPU_vocab_indices(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length, int len_W) {
	
	cudaSetDevice(ih_layer_info.device_number);

	this->h_input_vocab_indicies = h_input_vocab_indicies;
	this->current_length = current_length;
	this->h_input_vocab_indicies_Wgrad = h_input_vocab_indicies_Wgrad;

	CUDA_GET_LAST_ERROR("test1 d_vocab indicies prep LSTM layer");
	
	//transfer to the GPU
	cudaMemcpy(d_input_vocab_indicies, h_input_vocab_indicies, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer");
	
	cudaMemcpy(d_input_vocab_indicies_Wgrad, h_input_vocab_indicies_Wgrad, len_W*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer W_grad");
	w_grad_len = len_W;

	//Launch kernel to turn into 0/1's and indicies with no -1's
	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid = 128;
	vocab_to_01<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_01_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 1");

	vocab_to_nonM1<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 2");

	devSynchAll();
	setup_reverse_indicies<<<256,256>>>(d_reverse_unique_indicies,d_input_vocab_indicies_Wgrad,w_grad_len);
	CUDA_GET_LAST_ERROR("input setup reverse indicies");
	
	devSynchAll();
	//if(attent_layer!=NULL) {
	//	attent_layer->transfer_done = false; //??
	//}
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::prep_GPU_vocab_indices_bi(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length, int len_W) {
	
	cudaSetDevice(ih_layer_info.device_number);

	//reverse
	for(int i=0; i<current_length*minibatch_size; i++) {
		h_input_vocab_indicies_bi[i] = h_input_vocab_indicies[current_length*minibatch_size - i - 1];
	}
	
	for(int i=0; i<current_length; i++) {
		int low_index = IDX2C(0,i,minibatch_size);
		int high_index = IDX2C(minibatch_size-1,i,minibatch_size);
		while(low_index<=high_index) {
			int temp = h_input_vocab_indicies_bi[low_index];
			h_input_vocab_indicies_bi[low_index] = h_input_vocab_indicies_bi[high_index];
			h_input_vocab_indicies_bi[high_index] = temp;
			low_index++;
			high_index--;
		}
	}

	this->h_input_vocab_indicies_bi = h_input_vocab_indicies_bi;

}

/*
template<typename dType>
void Input_To_Hidden_Layer<dType>::prep_GPU_vocab_indices_bi(int *h_input_vocab_indicies_bi,int *h_input_vocab_indicies_Wgrad,int current_length, int len_W) {
	
	cudaSetDevice(ih_layer_info.device_number);

	//reverse
	for(int i=0; i<current_length*minibatch_size; i++) {
		h_input_vocab_indicies[i] = h_input_vocab_indicies_bi[current_length*minibatch_size - i - 1];
	}
	
	for(int i=0; i<current_length; i++) {
		int low_index = IDX2C(0,i,minibatch_size);
		int high_index = IDX2C(minibatch_size-1,i,minibatch_size);
		while(low_index<=high_index) {
			int temp = h_input_vocab_indicies[low_index];
			h_input_vocab_indicies[low_index] = h_input_vocab_indicies[high_index];
			h_input_vocab_indicies[high_index] = temp;
			low_index++;
			high_index--;
		}
	}

	this->h_input_vocab_indicies = h_input_vocab_indicies;
	this->current_length = current_length;
	this->h_input_vocab_indicies_Wgrad = h_input_vocab_indicies_Wgrad;

	CUDA_GET_LAST_ERROR("test1 d_vocab indicies prep LSTM layer");
	
	//transfer to the GPU
	cudaMemcpy(d_input_vocab_indicies, h_input_vocab_indicies, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer");
	
	cudaMemcpy(d_input_vocab_indicies_Wgrad, h_input_vocab_indicies_Wgrad, len_W*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer W_grad");
	w_grad_len = len_W;

	//Launch kernel to turn into 0/1's and indicies with no -1's
	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid = 128;
	vocab_to_01<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_01_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 1");

	vocab_to_nonM1<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 2");

	devSynchAll();
	setup_reverse_indicies<<<256,256>>>(d_reverse_unique_indicies,d_input_vocab_indicies_Wgrad,w_grad_len);
	CUDA_GET_LAST_ERROR("input setup reverse indicies");
	
	devSynchAll();
}

*/

template<typename dType>
void Input_To_Hidden_Layer<dType>::clear_gradients(bool init) {
	clear_gradients_GPU(init);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::clear_gradients_GPU(bool init) {
	
	cudaSetDevice(ih_layer_info.device_number);

	cudaDeviceSynchronize();
	cudaMemsetAsync(d_W_hi_grad, 0, LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s0);
	cudaMemsetAsync(d_b_i_grad, 0, LSTM_size*1*sizeof(dType),ih_layer_info.s1);

	cudaMemsetAsync(d_W_hf_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s2);
	cudaMemsetAsync(d_b_f_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s3);

	cudaMemsetAsync(d_W_hc_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s4);
	cudaMemsetAsync(d_b_c_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s5);

	cudaMemsetAsync(d_W_ho_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s6);
	cudaMemsetAsync(d_b_o_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s7);

	//CHANGE THIS TO NON NAIVE KERNEL
	if(init) {
		cudaMemset(d_small_W_grad,0,Embedding_size*minibatch_size*longest_sent*sizeof(dType)); //LSTM_size --> Embedding_size ?
		//cudaMemsetAsync(d_small_W_grad,0,Embedding_size*minibatch_size*longest_sent*sizeof(dType),ih_layer_info.s8);
	}
	else {
		cudaMemset(d_small_W_grad,0,Embedding_size*w_grad_len*sizeof(dType)); //
		//cudaMemsetAsync(d_small_W_grad,0,Embedding_size*w_grad_len*sizeof(dType),ih_layer_info.s8); //
	}

	cudaMemsetAsync(d_M_i_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s9);
	cudaMemsetAsync(d_M_f_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s10);
	cudaMemsetAsync(d_M_o_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s11);
	cudaMemsetAsync(d_M_c_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s12);

	if(feed_input) {
		cudaMemsetAsync(d_Q_i_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s9);
		cudaMemsetAsync(d_Q_f_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s10);
		cudaMemsetAsync(d_Q_o_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s11);
		cudaMemsetAsync(d_Q_c_grad,0,LSTM_size*Embedding_size*sizeof(dType),ih_layer_info.s12);
	}

	devSynchAll();
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::update_params(int time_index) {

	cudaSetDevice(ih_layer_info.device_number);
	cudaDeviceSynchronize();

	dType alpha = learning_rate;
	dType beta = 1;
	
	if(ADAM) {
		
		//for update
		//if (alpha_adam != learning_rate) {
		//	cout<<"new alpha_adam: "<<learning_rate<<endl; //for test
		//}
		//alpha_adam = learning_rate;
			
		int threads_per_block = 128;
		int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block, threads_per_block);
		
		//d_W_	
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s0>>>(d_W_hi, d_W_hi_grad, d_W_hi_mt, d_W_hi_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s2>>>(d_W_hf, d_W_hf_grad, d_W_hf_mt, d_W_hf_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s4>>>(d_W_hc, d_W_hc_grad, d_W_hc_mt, d_W_hc_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s6>>>(d_W_ho, d_W_ho_grad, d_W_ho_mt, d_W_ho_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		//d_M
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s9>>>(d_M_i, d_M_i_grad, d_M_i_mt, d_M_i_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s10>>>(d_M_f, d_M_f_grad, d_M_f_mt, d_M_f_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s12>>>(d_M_o, d_M_o_grad, d_M_o_mt, d_M_o_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s11>>>(d_M_c, d_M_c_grad, d_M_c_mt, d_M_c_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, LSTM_size);	
		
		if(feed_input) {

			update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s9>>>(d_Q_i, d_Q_i_grad, d_Q_i_mt, d_Q_i_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, Embedding_size);	
			
			update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s10>>>(d_Q_f, d_Q_f_grad, d_Q_f_mt, d_Q_f_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, Embedding_size);	
			
			update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s11>>>(d_Q_o, d_Q_o_grad, d_Q_o_mt, d_Q_o_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, Embedding_size);	
		
			update_params_adam<<<kernel,threads_per_block,0,ih_layer_info.s12>>>(d_Q_c, d_Q_c_grad, d_Q_c_mt, d_Q_c_vt, alpha_adam, beta_1, beta_2, epsilon, time_index, LSTM_size, Embedding_size);	
		
		}
		
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s1>>>(d_b_i,d_b_i_grad,d_b_i_mt,d_b_i_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_i");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s3>>>(d_b_f,d_b_f_grad,d_b_f_mt,d_b_f_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_f");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s5>>>(d_b_c,d_b_c_grad,d_b_c_mt,d_b_c_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_c");
		add_grad_vecs_adam<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s7>>>(d_b_o,d_b_o_grad,d_b_o_mt,d_b_o_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_o");
		


		//update word embedding
		if(!bi_dir && upper_layer.source_side){
			cublasSetStream(ih_layer_info.handle,ih_layer_info.s8);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,w_grad_len,&alpha,d_small_W_grad,LSTM_size,
				&beta,model->input_layer_source_bi.d_small_W_grad,LSTM_size,d_small_W_grad,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

			update_sparse_grad_adam<<<256,256,0,ih_layer_info.s8>>>(d_W,d_small_W_grad,d_W_mt,d_W_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,d_input_vocab_indicies_Wgrad,w_grad_len,Embedding_size); 
		}
		
		//target
		if(!bi_dir && !upper_layer.source_side){
			update_sparse_grad_adam<<<256,256,0,ih_layer_info.s8>>>(d_W,d_small_W_grad,d_W_mt,d_W_vt,alpha_adam,beta_1,beta_2,epsilon,time_index,d_input_vocab_indicies_Wgrad,w_grad_len,Embedding_size); 
		}

	
	}
	else { // SGD
		//normal matrices
		cublasSetStream(ih_layer_info.handle,ih_layer_info.s0);
		//d_W_hi = alpha * d_W_hi_grad + d_W_hi
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hi_grad, LSTM_size, &beta, d_W_hi, LSTM_size, d_W_hi, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s2);
		//d_W_hf = alpha * d_W_hf_grad + d_W_hf
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hf_grad, LSTM_size, &beta, d_W_hf, LSTM_size, d_W_hf, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s4);
		//d_W_hc = alpha * d_W_hc_grad + d_W_hc
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_hc_grad, LSTM_size, &beta, d_W_hc, LSTM_size, d_W_hc, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s6);
		//d_W_ho = alpha * d_W_ho_grad + d_W_ho
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
			d_W_ho_grad, LSTM_size, &beta, d_W_ho, LSTM_size, d_W_ho, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s9);
		//d_M_i += alpha * d_M_i_grad
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
			d_M_i_grad, LSTM_size, &beta, d_M_i, LSTM_size, d_M_i, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s10);
		//d_M_f += alpha * d_M_f_grad 
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
			d_M_f_grad, LSTM_size, &beta, d_M_f, LSTM_size, d_M_f, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s12);
		//d_M_c += alpha * d_M_c_grad
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
			d_M_c_grad, LSTM_size, &beta, d_M_c, LSTM_size, d_M_c, LSTM_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(ih_layer_info.handle,ih_layer_info.s11);
		//d_M_o += alpha * d_M_o_grad
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
			d_M_o_grad, LSTM_size, &beta, d_M_o, LSTM_size, d_M_o, LSTM_size),"CUBLAS addition update parameter failed\n");
		
		//cout<<"update show d_M_i: "<<endl;
		//show_matrix(d_M_i, LSTM_size, Embedding_size);

		if(feed_input) {

			cublasSetStream(ih_layer_info.handle,ih_layer_info.s9);
			//d_Q_i += alpha * d_Q_i_grad
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
				d_Q_i_grad, LSTM_size, &beta, d_Q_i, LSTM_size, d_Q_i, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(ih_layer_info.handle,ih_layer_info.s10);
			//d_Q_f += alpha * d_Q_f_grad
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
				d_Q_f_grad, LSTM_size, &beta, d_Q_f, LSTM_size, d_Q_f, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(ih_layer_info.handle,ih_layer_info.s12);
			//d_Q_c += alpha * d_Q_c_grad
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
				d_Q_c_grad, LSTM_size, &beta, d_Q_c, LSTM_size, d_Q_c, LSTM_size),"CUBLAS addition update parameter failed\n");

			cublasSetStream(ih_layer_info.handle,ih_layer_info.s11);
			//d_Q_o += alpha * d_Q_o_grad
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, Embedding_size, &alpha, 
				d_Q_o_grad, LSTM_size, &beta, d_Q_o, LSTM_size, d_Q_o, LSTM_size),"CUBLAS addition update parameter failed\n");

		}

		
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s1>>>(d_b_i,d_b_i_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_i");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s3>>>(d_b_f,d_b_f_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_f");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s5>>>(d_b_c,d_b_c_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_c");
		add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s7>>>(d_b_o,d_b_o_grad,learning_rate,LSTM_size*1);
		CUDA_GET_LAST_ERROR("update d_b_o");
		
		//update word embedding
		if(!bi_dir && upper_layer.source_side){
			cublasSetStream(ih_layer_info.handle,ih_layer_info.s8);
			CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,w_grad_len,&alpha,d_small_W_grad,LSTM_size,
				&beta,model->input_layer_source_bi.d_small_W_grad,LSTM_size,d_small_W_grad,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

			update_sparse_grad<<<256,256,0,ih_layer_info.s8>>>(d_W,d_small_W_grad,d_input_vocab_indicies_Wgrad,w_grad_len,learning_rate,Embedding_size); // LSTM_size -> Embedding_size ?
		}
		
		//target
		if(!bi_dir && !upper_layer.source_side){
			update_sparse_grad<<<256,256,0,ih_layer_info.s8>>>(d_W,d_small_W_grad,d_input_vocab_indicies_Wgrad,w_grad_len,learning_rate,Embedding_size); // LSTM_size -> Embedding_size ?
		}

	}
	
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::scale_gradients() {

	cudaSetDevice(ih_layer_info.device_number);
	scale_functor unary_op(minibatch_size);

	thrust::for_each(thrust_d_W_hi_grad,thrust_d_W_hi_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_i_grad,thrust_d_b_i_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hf_grad,thrust_d_W_hf_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_f_grad,thrust_d_b_f_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hc_grad,thrust_d_W_hc_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_c_grad,thrust_d_b_c_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_ho_grad,thrust_d_W_ho_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_o_grad,thrust_d_b_o_grad + LSTM_size*1,unary_op);

	
	thrust::for_each(thrust_d_small_W_grad,thrust_d_small_W_grad+Embedding_size*w_grad_len,unary_op);

	thrust::for_each(thrust_d_M_i_grad,thrust_d_M_i_grad + LSTM_size*Embedding_size,unary_op);
	thrust::for_each(thrust_d_M_f_grad,thrust_d_M_f_grad + LSTM_size*Embedding_size,unary_op);
	thrust::for_each(thrust_d_M_o_grad,thrust_d_M_o_grad + LSTM_size*Embedding_size,unary_op);
	thrust::for_each(thrust_d_M_c_grad,thrust_d_M_c_grad + LSTM_size*Embedding_size,unary_op);

	if(feed_input) {
		thrust::for_each(thrust_d_Q_i_grad,thrust_d_Q_i_grad + LSTM_size*Embedding_size,unary_op);
		thrust::for_each(thrust_d_Q_f_grad,thrust_d_Q_f_grad + LSTM_size*Embedding_size,unary_op);
		thrust::for_each(thrust_d_Q_o_grad,thrust_d_Q_o_grad + LSTM_size*Embedding_size,unary_op);
		thrust::for_each(thrust_d_Q_c_grad,thrust_d_Q_c_grad + LSTM_size*Embedding_size,unary_op);
	}
	
	devSynchAll();
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::update_weights(int time_index) {
	update_weights_GPU(time_index);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::update_weights_GPU(int time_index) {

	cudaSetDevice(ih_layer_info.device_number);

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

		norm_clip_GPU_v2(thrust_d_small_W_grad,d_small_W_grad,norm_clip,Embedding_size*w_grad_len,d_temp_result,d_result);
	
		norm_clip_GPU_v2(thrust_d_M_i_grad,d_M_i_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_f_grad,d_M_f_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_o_grad,d_M_o_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_c_grad,d_M_c_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);

		if(feed_input) {
			norm_clip_GPU_v2(thrust_d_Q_i_grad,d_Q_i_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_Q_f_grad,d_Q_f_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_Q_o_grad,d_Q_o_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_Q_c_grad,d_Q_c_grad,norm_clip,LSTM_size*Embedding_size,d_temp_result,d_result);
		}
	
	}

	update_params(time_index);

	devSynchAll();
	
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::dump_weights_GPU(ofstream &output) {

	cudaSetDevice(ih_layer_info.device_number);

	write_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_i,LSTM_size,1,output);

	write_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_f,LSTM_size,1,output);

	write_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_c,LSTM_size,1,output);

	write_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_o,LSTM_size,1,output);
	
	if(!bi_dir){
		write_matrix_GPU(d_W,Embedding_size,input_vocab_size,output);
	}

	write_matrix_GPU(d_M_i,LSTM_size,Embedding_size,output);
	write_matrix_GPU(d_M_f,LSTM_size,Embedding_size,output);
	write_matrix_GPU(d_M_o,LSTM_size,Embedding_size,output);
	write_matrix_GPU(d_M_c,LSTM_size,Embedding_size,output);

	if(feed_input) {
		write_matrix_GPU(d_Q_i,LSTM_size,Embedding_size,output);
		write_matrix_GPU(d_Q_f,LSTM_size,Embedding_size,output);
		write_matrix_GPU(d_Q_o,LSTM_size,Embedding_size,output);
		write_matrix_GPU(d_Q_c,LSTM_size,Embedding_size,output);
	}


}

template<typename dType>
void Input_To_Hidden_Layer<dType>::dump_weights(std::ofstream &output) {
	dump_weights_GPU(output);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::load_weights_GPU(ifstream &input) {

	cudaSetDevice(ih_layer_info.device_number);

	read_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_i,LSTM_size,1,input);

	read_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);

	read_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);

	read_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);
	
	if(!bi_dir){
		read_matrix_GPU(d_W,Embedding_size,input_vocab_size,input);
	}

	read_matrix_GPU(d_M_i,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_f,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_o,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_c,LSTM_size,Embedding_size,input);


	if(feed_input) {
		read_matrix_GPU(d_Q_i,LSTM_size,Embedding_size,input);
		read_matrix_GPU(d_Q_f,LSTM_size,Embedding_size,input);
		read_matrix_GPU(d_Q_o,LSTM_size,Embedding_size,input);
		read_matrix_GPU(d_Q_c,LSTM_size,Embedding_size,input);
	}
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::load_weights(ifstream &input) {
	load_weights_GPU(input);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_all_gradients(dType epsilon) {
	check_all_gradients_GPU(epsilon);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_all_gradients_GPU(dType epsilon) {

	cudaSetDevice(ih_layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR INPUT LAYER GPU-------------------------\n";
	
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
	*/
	std::cout << "GRADIENT CHECKING FOR b_o\n";
	check_gradient_GPU(epsilon,d_b_o,d_b_o_grad,LSTM_size,1);
	
	//std::cout << "GRADIENT CHECKING FOR M_i\n";
	//check_gradient_GPU(epsilon,d_M_i,d_M_i_grad,LSTM_size,Embedding_size);
	
	std::cout << "GRADIENT CHECKING FOR M_f\n";
	check_gradient_GPU(epsilon,d_M_f,d_M_f_grad,LSTM_size,Embedding_size);
	/*
	std::cout << "GRADIENT CHECKING FOR M_o\n";
	check_gradient_GPU(epsilon,d_M_o,d_M_o_grad,LSTM_size,Embedding_size);
	
	std::cout << "GRADIENT CHECKING FOR M_c\n";
	check_gradient_GPU(epsilon,d_M_c,d_M_c_grad,LSTM_size,Embedding_size);

	if(feed_input) {
		std::cout << "GRADIENT CHECKING FOR Q_i\n";
		check_gradient_GPU(epsilon,d_Q_i,d_Q_i_grad,LSTM_size,Embedding_size);
		
		std::cout << "GRADIENT CHECKING FOR Q_f\n";
		check_gradient_GPU(epsilon,d_Q_f,d_Q_f_grad,LSTM_size,Embedding_size);

		std::cout << "GRADIENT CHECKING FOR Q_o\n";
		check_gradient_GPU(epsilon,d_Q_o,d_Q_o_grad,LSTM_size,Embedding_size);
		
		std::cout << "GRADIENT CHECKING FOR Q_c\n";
		check_gradient_GPU(epsilon,d_Q_c,d_Q_c_grad,LSTM_size,Embedding_size);
	}
	*/
	
	//
	if(!bi_dir && upper_layer.source_side) {
		std::cout << "GRADIENT CHECKING FOR W SMALL\n";
		check_gradient_GPU_SPARSE_bi(epsilon,d_W,d_small_W_grad,model->input_layer_source_bi.d_small_W_grad,Embedding_size,h_input_vocab_indicies_Wgrad,w_grad_len);
	}
	
	//target	
	if(!upper_layer.source_side) {
		std::cout << "GRADIENT CHECKING FOR W SMALL\n";
		check_gradient_GPU_SPARSE(epsilon,d_W,d_small_W_grad,Embedding_size,h_input_vocab_indicies_Wgrad,w_grad_len);
	}
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

	cudaSetDevice(ih_layer_info.device_number);
	
	//cout<<"test 1 check gradient"<<endl;

	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	
	//cout<<"test 2 check gradient"<<endl;
	
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);

	//cout<<"test 2.1 check gradient: "<<d_thrust_grad[0]<<endl;
	
	for(int i=0; i<min(5,rows); i++) {
		for(int j=0; j<min(2,cols); j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			
			//cout<<"test 3 check gradient: "<<d_thrust_mat[IDX2C(i,j,rows)]<<endl;
			//cout<<"test 3 check gradient"<<endl;
			
			loss = model->getError();
			
			//cout<<"test 4 check gradient"<<endl;
			
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError();
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			//std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
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


template<typename dType>
void Input_To_Hidden_Layer<dType>::check_gradient_GPU_SPARSE(dType epsilon,dType *d_mat,dType *d_grad,int LSTM_size,int *h_unique_indicies,int curr_num_unique) {
	cudaSetDevice(ih_layer_info.device_number);
	cudaDeviceSynchronize();
	
	//cout<<"show d_grad"<<endl;
	//show_matrix(d_grad, LSTM_size, curr_num_unique);
	
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<min(5,curr_num_unique); i++) {
		for(int j=0; j<min(2,LSTM_size); j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			loss = model->getError();
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= -2*epsilon;
			loss -=model->getError();
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_gradient_GPU_SPARSE_bi(dType epsilon,dType *d_mat,dType *d_grad,dType *d_grad_2,int LSTM_size,int *h_unique_indicies,int curr_num_unique) {
	cudaSetDevice(ih_layer_info.device_number);
	cudaDeviceSynchronize();
	
	//cout<<"show d_grad"<<endl;
	//show_matrix(d_grad, LSTM_size, curr_num_unique);
	
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	thrust::device_ptr<dType> d_thrust_grad_2 = thrust::device_pointer_cast(d_grad_2);
	for(int i=0; i<min(5,curr_num_unique); i++) {
		for(int j=0; j<min(2,LSTM_size); j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			loss = model->getError();
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= -2*epsilon;
			loss -=model->getError();
			cudaSetDevice(ih_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]+d_thrust_grad_2[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}



