
//#include "Input_To_Hidden_Layer.h"

void Input_To_Hidden_Layer::init_Input_To_Hidden_Layer(int Embedding_size, int LSTM_size, int minibatch_size, int vocab_size, bool feed_input, int gpu_num, bool bi_dir) {
	
	this->Embedding_size = Embedding_size;
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->vocab_size = vocab_size;
	this->feed_input = feed_input;
	this->gpu_num = gpu_num;
	this->bi_dir = bi_dir; 
	
	gpu_info.init(gpu_num);
	init_params();	

}

void Input_To_Hidden_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);
	read_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_i,LSTM_size,1,input);
	
	
	read_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);

	read_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);

	read_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);
	
	if(!bi_dir) {
		read_matrix_GPU(d_W,LSTM_size,vocab_size,input); // lookup table

		//cout<<"show d_W: "<<endl;
		//show_matrix(d_W,LSTM_size,vocab_size);
	}
	//else {
	//	d_W = model->
	//}
	//i,f,o,c
	read_matrix_GPU(d_M_i,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_f,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_o,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_M_c,LSTM_size,Embedding_size,input);
		
	//cout<<"show d_M_i: "<<endl;
	//show_matrix(d_M_i,LSTM_size,Embedding_size);

	if(feed_input) {
		load_weight_feed_input(input);
	}
	
}

void Input_To_Hidden_Layer::load_weight_feed_input(ifstream &input) {

	//feed_input i,f,o,c
	cudaSetDevice(gpu_num);
	read_matrix_GPU(d_Q_i,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_Q_f,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_Q_o,LSTM_size,Embedding_size,input);
	read_matrix_GPU(d_Q_c,LSTM_size,Embedding_size,input);
}

void Input_To_Hidden_Layer::init_params() {
	
	cudaSetDevice(gpu_num);
	cudaMalloc((void**) &d_W_hi, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_hf, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_hc, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_ho, LSTM_size*LSTM_size*sizeof(float));
	
	cudaMalloc((void**) &d_b_i, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_f, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_c, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_o, LSTM_size*1*sizeof(float));
	
	cudaMalloc((void**) &d_W, LSTM_size*vocab_size*sizeof(float));

	cudaMalloc((void**) &d_M_i, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_M_f, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_M_c, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_M_o, LSTM_size*Embedding_size*sizeof(float));

	cudaMalloc((void**)&d_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_2, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_3, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_4, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_5, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_6, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_7, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_8, LSTM_size*minibatch_size*sizeof(float));

	if(feed_input) {
		init_param_feed_input();
	}

	//node
	cudaMalloc((void**)&d_wid, minibatch_size*sizeof(int));
	cudaMalloc((void**)&d_x_t, Embedding_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_i_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_f_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_prime_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_o_t, LSTM_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_init_hidden_vector, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_init_cell_vector, LSTM_size*minibatch_size*sizeof(float));
	cudaMemset(d_init_hidden_vector,0,LSTM_size*minibatch_size*sizeof(float));
	cudaMemset(d_init_cell_vector,0,LSTM_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_h_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_h_t_prev, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t_prev, LSTM_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_h_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float)); // for prepare_forward_decode
	cudaMalloc((void**)&d_c_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float)); //
	
	cudaMalloc((void**)&d_h_t_feed, Embedding_size*minibatch_size*sizeof(float));

	cudaMalloc((void**)&d_father_idx, minibatch_size*sizeof(int));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Input_To_Hidden_Layer::init_param_feed_input() {
	
	cudaSetDevice(gpu_num);
	cudaMalloc((void**) &d_Q_i, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_Q_f, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_Q_c, LSTM_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_Q_o, LSTM_size*Embedding_size*sizeof(float));

	cudaMalloc((void**)&d_temp_1_feed, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_2_feed, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_3_feed, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_4_feed, LSTM_size*minibatch_size*sizeof(float));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}


void Input_To_Hidden_Layer::prepare_forward(int *input_wids, float *d_h_t_prev, float *d_c_t_prev) {

	this->input_wids = input_wids;
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;

}

void Input_To_Hidden_Layer::prepare_forward_decode(int *tgt_wids, int *father_idx, int B, float *d_final_temp_2,  cudaEvent_t &prev_event) {

	this->input_wids = tgt_wids;
	cudaMemcpy(d_father_idx, father_idx, B*sizeof(int), cudaMemcpyHostToDevice);

	dim3 block_shape(256,1,1);
	dim3 grid_shape(B,(LSTM_size + block_shape.x - 1)/block_shape.x,1);
	cudaStreamWaitEvent(gpu_info.s0, prev_event, 0);

	//cout<<"show Input_To_Hidden_Layer d_h_t **: "<<endl;
	//show_matrix(d_h_t, LSTM_size, B);

	//cout<<"show Input_To_Hidden_Layer d_father_idx **: "<<endl;
	//show_matrix(d_father_idx, B, 1);
	
	lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_h_t_prev_tmp,d_h_t,d_father_idx,LSTM_size);
	lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_c_t_prev_tmp,d_c_t,d_father_idx,LSTM_size);
	
	this->d_h_t_prev = d_h_t_prev_tmp;	//?
	this->d_c_t_prev = d_c_t_prev_tmp;	

	//cout<<"show Input_To_Hidden_Layer d_h_t_prev: "<<endl;
	//show_matrix(d_h_t_prev, LSTM_size, B);
	
	//cout<<"show Input_To_Hidden_Layer d_c_t_prev: "<<endl;
	//show_matrix(d_c_t_prev, LSTM_size, B);

	//cout<<"show Input_To_Hidden_Layer d_c_t_prev_1: "<<endl;
	//show_matrix(d_c_t_prev_1, LSTM_size, B);
	
	if(feed_input) {
		lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_h_t_feed,d_final_temp_2,d_father_idx,Embedding_size);  //
	}
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Input_To_Hidden_Layer::forward_prop(int index, int T, int B) {
	
	minibatch_size = B;

	cudaSetDevice(gpu_num);
	dim3 block_shape(256,1,1);
	dim3 grid_shape(minibatch_size, (Embedding_size + block_shape.x -1)/block_shape.x, 1); //(1 4 1)
	cudaMemcpy(d_wid, input_wids, minibatch_size*sizeof(int), cudaMemcpyHostToDevice);

	lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_x_t, d_W, d_wid, Embedding_size); //(1000 1)
	cudaEventRecord(gpu_info.e0, gpu_info.s0);
	
	//cout<<"*********************Input_To_Hidden_Layer show d_x_t:*** "<<endl;
	//show_matrix(d_W, Embedding_size, vocab_size);
	
	//cout<<"*********************Input_To_Hidden_Layer show d_x_t:*** "<<endl;
	//show_matrix(d_x_t, Embedding_size, minibatch_size);
	
	float alpha = 1.0;
	float beta = 0.0;

	//d_M_ * d_x_t
	cublasSetStream(gpu_info.handle, gpu_info.s1);
	cudaStreamWaitEvent(gpu_info.s1, gpu_info.e0, 0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_M_i,LSTM_size,d_x_t,Embedding_size,&beta,d_temp_1,LSTM_size),"error");// (1000, 1000), (1000,1) , (1000,1)
	cudaEventRecord(gpu_info.e1, gpu_info.s1);

	cublasSetStream(gpu_info.handle, gpu_info.s2);
	cudaStreamWaitEvent(gpu_info.s2, gpu_info.e0, 0);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_M_f,LSTM_size,d_x_t,Embedding_size,&beta,d_temp_3,Embedding_size);// (H, E), (E,1) , ((H,1)
	cudaEventRecord(gpu_info.e2, gpu_info.s2);
	
	cublasSetStream(gpu_info.handle, gpu_info.s3);
	cudaStreamWaitEvent(gpu_info.s3, gpu_info.e0, 0);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_M_c,LSTM_size,d_x_t,Embedding_size,&beta,d_temp_5,Embedding_size); 
	cudaEventRecord(gpu_info.e3, gpu_info.s3);
	
	cublasSetStream(gpu_info.handle, gpu_info.s4);
	cudaStreamWaitEvent(gpu_info.s4, gpu_info.e0, 0);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_o,LSTM_size,d_x_t,Embedding_size,&beta,d_temp_7,Embedding_size);
	cudaEventRecord(gpu_info.e4,gpu_info.s4);

	//cout<<"*********************Input_To_Hidden_Layer show d_temp_7:*** "<<endl;
	//show_matrix(d_temp_7, LSTM_size, minibatch_size);

	//d_Q_ * d_x_tild
	if(feed_input && index != 0) {
	
		cublasSetStream(gpu_info.handle, gpu_info.s1_feed);
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_Q_i,LSTM_size,d_h_t_feed,Embedding_size,&beta,d_temp_1_feed,LSTM_size);// (1000, 1000), (1000,1) , (1000,1)
		cudaEventRecord(gpu_info.e1_feed, gpu_info.s1_feed);
	
		cublasSetStream(gpu_info.handle, gpu_info.s2_feed);
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_Q_f,LSTM_size,d_h_t_feed,Embedding_size,&beta,d_temp_2_feed,LSTM_size);// (1000, 1000), (1000,B) , (1000,B)
		cudaEventRecord(gpu_info.e2_feed, gpu_info.s2_feed);
	
		cublasSetStream(gpu_info.handle, gpu_info.s3_feed);
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_Q_c,LSTM_size,d_h_t_feed,Embedding_size,&beta,d_temp_3_feed,LSTM_size);
		cudaEventRecord(gpu_info.e3_feed, gpu_info.s3_feed);
	
		cublasSetStream(gpu_info.handle, gpu_info.s4_feed);
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,Embedding_size,&alpha,d_Q_o,LSTM_size,d_h_t_feed,Embedding_size,&beta,d_temp_4_feed,LSTM_size);
		cudaEventRecord(gpu_info.e4_feed, gpu_info.s4_feed);

		//cout<<"*********************Input_To_Hidden_Layer show d_temp_4_feed:*** "<<endl;
		//show_matrix(d_temp_4_feed, LSTM_size, minibatch_size);
	}


	// d_W_ * h_{t-1}
	cublasSetStream(gpu_info.handle, gpu_info.s5);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hi,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_2,LSTM_size); // (1000, 1000), (1000, 1), (1000,1)
	
	cublasSetStream(gpu_info.handle, gpu_info.s6);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hf,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_4,LSTM_size); // (H,H), (H,B), (H,B)
	
	cublasSetStream(gpu_info.handle, gpu_info.s7);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hc,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_6,LSTM_size); 
	
	cublasSetStream(gpu_info.handle, gpu_info.s8);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_ho,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_8,LSTM_size); 

	//cout<<"*********************Input_To_Hidden_Layer show d_temp_8:*** "<<endl;
	//show_matrix(d_temp_8, LSTM_size, minibatch_size);
	

	// gate
	dim3 block_shape1(256,1,1);
	dim3 grid_shape1((LSTM_size + block_shape1.x - 1)/block_shape1.x,minibatch_size,1); // (4,minibatch_size,1)
	

	if (!feed_input || index == 0){

		cudaStreamWaitEvent(gpu_info.s5, gpu_info.e1, 0);
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s5>>>(d_i_t, d_temp_1, d_temp_2, d_b_i, LSTM_size); //d_i_t   (1000,1)
		cudaEventRecord(gpu_info.e5, gpu_info.s5);

		cudaStreamWaitEvent(gpu_info.s6, gpu_info.e2, 0);
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s6>>>(d_f_t, d_temp_3, d_temp_4, d_b_f, LSTM_size); //d_f_t
		cudaEventRecord(gpu_info.e6, gpu_info.s6);
		
		cudaStreamWaitEvent(gpu_info.s7, gpu_info.e3, 0);
		forward_tanh_kernel<<<grid_shape1,block_shape1,0,gpu_info.s7>>>(d_c_prime_t, d_temp_5, d_temp_6, d_b_c, LSTM_size); //d_c_prime_t
		cudaEventRecord(gpu_info.e7, gpu_info.s7);
		
		cudaStreamWaitEvent(gpu_info.s8, gpu_info.e4, 0);
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s8>>>(d_o_t, d_temp_7, d_temp_8, d_b_o, LSTM_size); //d_o_t
		cudaEventRecord(gpu_info.e8, gpu_info.s8);
	}
	else {

		cudaStreamWaitEvent(gpu_info.s5, gpu_info.e1, 0);
		cudaStreamWaitEvent(gpu_info.s5, gpu_info.e1_feed, 0);
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s5>>>(d_i_t, d_temp_1, d_temp_2, d_temp_1_feed, d_b_i, LSTM_size); //d_i_t   (1000,1)
		cudaEventRecord(gpu_info.e5, gpu_info.s5);
		
		cudaStreamWaitEvent(gpu_info.s6, gpu_info.e2, 0);
		cudaStreamWaitEvent(gpu_info.s6, gpu_info.e2_feed, 0);
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s6>>>(d_f_t, d_temp_3, d_temp_4, d_temp_2_feed, d_b_f, LSTM_size); //d_f_t
		cudaEventRecord(gpu_info.e6, gpu_info.s6);
		
		cudaStreamWaitEvent(gpu_info.s7, gpu_info.e3, 0);
		cudaStreamWaitEvent(gpu_info.s7, gpu_info.e3_feed, 0);
		forward_tanh_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s7>>>(d_c_prime_t, d_temp_5, d_temp_6, d_temp_3_feed, d_b_c, LSTM_size); //d_c_prime_t
		cudaEventRecord(gpu_info.e7, gpu_info.s7);
		
		cudaStreamWaitEvent(gpu_info.s8, gpu_info.e4, 0);
		cudaStreamWaitEvent(gpu_info.s8, gpu_info.e4_feed, 0);
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s8>>>(d_o_t, d_temp_7, d_temp_8, d_temp_4_feed, d_b_o, LSTM_size); //d_o_t
		cudaEventRecord(gpu_info.e8, gpu_info.s8);
	}
	
	//cout<<"*********************Input_To_Hidden_Layer show d_o_t:*** "<<endl;
	//show_matrix(d_o_t, LSTM_size, minibatch_size);

	// c_t
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e5, 0);
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e6, 0);
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e7, 0);
	forward_c_t_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_c_t, d_f_t, d_c_t_prev, d_i_t, d_c_prime_t, LSTM_size);  
	
	// h_t
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e8, 0);
	forward_h_t_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_h_t, d_o_t, d_c_t, LSTM_size); 


	//copy d_h_t to upper_layer
	if(bi_dir) {
		//if(index == 1) {	
			//cout<<"Input_To_Hidden_Layer bi_dir show d_h_t:*** "<<endl;
			//show_matrix(d_h_t, LSTM_size, minibatch_size);
			//string a;
			//cin>>a;
		//}
		cudaMemcpy(upper_layer.hidden_layer->d_h_t_below_bi+(T-1-index)*LSTM_size, d_h_t, LSTM_size*sizeof(float), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(upper_layer.hidden_layer->d_below_bi_mat[0], d_h_t, LSTM_size*sizeof(float), cudaMemcpyDeviceToDevice);
		//upper_layer.hidden_layer->d_below_bi_mat[0] = d_h_t;	
		//upper_layer.hidden_layer->d_h_t_below_bi = d_h_t;
	}
	else {
		
		//cout<<"Input_To_Hidden_Layer single_dir show d_h_t:*** "<<endl;
		//show_matrix(d_h_t, LSTM_size, minibatch_size);
		//string a;
		//cin>>a;
		
		upper_layer.hidden_layer->d_h_t_below = d_h_t;
	}
	
	cudaEventRecord(gpu_info.h_t_below_transfer, gpu_info.s0);

	//cout<<"Input_To_Hidden_Layer show d_h_t:*** "<<endl;
	//show_matrix(d_h_t, LSTM_size, minibatch_size);

}




