
//#include "Attention_Softmax_Layer.h"

void Attention_Softmax_Layer::init_Attention_Softmax_Layer(int Embedding_size, int LSTM_size, int minibatch_size, int vocab_size, int gpu_num) {

	this->Embedding_size = Embedding_size;
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->vocab_size = vocab_size;
	this->gpu_num = gpu_num;

	gpu_info.init(gpu_num);
	
	init_params();

}

void Attention_Softmax_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);

	// attention params
	read_matrix_GPU(d_output_bias,Embedding_size,1,input);
	read_matrix_GPU(d_W_c_p1,Embedding_size,LSTM_size,input);
	read_matrix_GPU(d_W_c_p2,Embedding_size,LSTM_size,input);
   	
	// softmax params
	read_matrix_GPU(d_D,vocab_size,Embedding_size,input);
	read_matrix_GPU(d_b_d,vocab_size,1,input);
	
	//cout<<"show d_D: "<<endl;
	//show_matrix(d_D, vocab_size, Embedding_size);
	
	

}

void Attention_Softmax_Layer::init_params() {
	
	cudaSetDevice(gpu_num);
	
	cudaMalloc((void**) &d_output_bias, Embedding_size*1*sizeof(float));
	cudaMalloc((void**) &d_W_c_p1, Embedding_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_c_p2, Embedding_size*LSTM_size*sizeof(float));

	cudaMalloc((void**)&d_final_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_final_temp_2, LSTM_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**) &d_D, vocab_size*Embedding_size*sizeof(float));
	cudaMalloc((void**) &d_b_d, vocab_size*1*sizeof(float));

	//node
	cudaMalloc((void**)&d_h_t_below, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_h_t_source, LSTM_size*200*sizeof(float)); // source maximum length 200

	cudaMalloc((void**)&d_alignment, minibatch_size*200*sizeof(float)); //
	cudaMalloc((void**)&d_normal_alignment, minibatch_size*200*sizeof(float)); //

	cudaMalloc((void**)&d_c_att_t, LSTM_size*minibatch_size*sizeof(float));
	
	vector<float> ones(vocab_size,1);
	cudaMalloc((void**)&d_ones, vocab_size*sizeof(float));
	cudaMemcpy(d_ones, &ones[0], vocab_size*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_outputdist, vocab_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_outputdist_sum, minibatch_size*sizeof(float)); 
	cudaMalloc((void**)&d_logit_softmax, vocab_size*minibatch_size*sizeof(float));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Attention_Softmax_Layer::attention_softmax_forward_prop(int T, int B) {

	//******************************attention****************************//
	
	cudaSetDevice(gpu_num);
	cudaStreamWaitEvent(gpu_info.s0, lower_layer.hidden_layer->gpu_info.h_t_below_transfer, 0);
	
	float alpha = 1.0;
	float beta = 0.0;


	//alignment_kernel
	//cudaMemcpy(d_h2_below, d_h2_t, LSTM_size*B*sizeof(float), cudaMemcpyDeviceToDevice); // test
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,B,T,LSTM_size,&alpha,d_h_t_below,LSTM_size,d_h_t_source,LSTM_size,&beta,d_alignment,B),"alignment_kernel error"); // (1000,B).T , (1000,T) -> d_alignment (B,T)
	
	//norm  (B,T)
	normalization_alignment_kernel<<<1,B,0,gpu_info.s0>>>(d_normal_alignment,d_alignment,B,T);
	
	// sum(attention*d_h2_source)
	// (LSTM_size,T) * (B,T).T -> d_c_t (LSTM_size,B)
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	//!!
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,B,T,&alpha,d_h_t_source,LSTM_size,d_normal_alignment,B,&beta,d_c_att_t,LSTM_size),"attention sum error");
	
	//d_final_temp_1 = W_c_p1 * d_c_att_t
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,Embedding_size,B,LSTM_size,&alpha,d_W_c_p1,Embedding_size,d_c_att_t,LSTM_size,&beta,d_final_temp_1,Embedding_size),"d_final_temp_1 error");

	//d_final_temp_2 = W_c_p2 * d_h_att_t(d_h_t_below)
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,Embedding_size,B,LSTM_size,&alpha,d_W_c_p2,Embedding_size,d_h_t_below,LSTM_size,&beta,d_final_temp_2,Embedding_size),"d_final_temp_2 error");

	//add in the bias and tanh
	tanh_att_forward_kernel<<<std::min(256,(Embedding_size*B + 256 - 1)/256),256,0,gpu_info.s0>>>(d_final_temp_2, d_final_temp_1, d_final_temp_2, d_output_bias, Embedding_size, B);    // get h_t_mat
		

	//******************************softmax****************************//
	
	//d_outputdist = d_D * d_h_t_mat // (target_vocab_size,LSTM_size) * (LSTM_size,B)
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,vocab_size,B,Embedding_size,&alpha,d_D,vocab_size,d_final_temp_2,Embedding_size,&beta,d_outputdist,vocab_size),"d_outputdist error");
	
	//d_outputdist = d_outputdist + d_b_d
	dim3 block_shape2(256,1,1);
	dim3 grid_shape2((vocab_size+256-1)/256,B,1);
	matrix_bias_kernel<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(d_outputdist,d_b_d,d_outputdist,vocab_size);  // (target_vocab_size,B)
	

	exp_overflow_prevention<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(d_outputdist,vocab_size); //exp(x-10)
	
	//cout<<"d_outputdist: "<<endl;
	//show_matrix(d_outputdist, vocab_size, B);
	
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	CUBLAS_ERROR_WRAPPER(cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,1,B,vocab_size,&alpha,d_ones,1,d_outputdist,vocab_size,&beta,d_outputdist_sum,1),"d_outputdist_sum error");
	
	divide<<<grid_shape2,block_shape2,0,gpu_info.s0>>>(d_logit_softmax,d_outputdist,d_outputdist_sum,vocab_size);
	
	//cout<<"d_logit_softmax: "<<endl;
	//show_matrix(d_logit_softmax, vocab_size, B);
	
	//string mytest;
	//cin>>mytest;

	cudaEventRecord(gpu_info.e0, gpu_info.s0);


}



