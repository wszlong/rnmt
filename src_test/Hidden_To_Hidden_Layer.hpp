
//#include "Hidden_To_Hidden_Layer.h"

void Hidden_To_Hidden_Layer::init_Hidden_To_Hidden_Layer(int LSTM_size, int minibatch_size, int gpu_num) {
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->gpu_num = gpu_num;

	gpu_info.init(gpu_num);
	init_params();
	
}

void Hidden_To_Hidden_Layer::load_weight(ifstream &input) {
	
	cudaSetDevice(gpu_num);
	read_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_i,LSTM_size,1,input);

	read_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);

	read_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);

	read_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);

	//i,f,o,c
	read_matrix_GPU(d_M_i,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_f,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_o,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_c,LSTM_size,LSTM_size,input);
	
	//cout<<lower_layer.lower_input<<" "<<upper_layer.source_side<<endl;
	if(lower_layer.lower_input && upper_layer.source_side) {
		read_matrix_GPU(d_U_i,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_f,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_o,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_U_c,LSTM_size,LSTM_size,input);
		
		//cout<<"show d_U_i test: "<<endl;
		//show_matrix(d_U_i, LSTM_size, LSTM_size);
	}

}

void Hidden_To_Hidden_Layer::init_params() {
	
	cudaSetDevice(gpu_num);
	cudaMalloc((void**) &d_W_hi, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_hf, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_hc, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_W_ho, LSTM_size*LSTM_size*sizeof(float));
	
	cudaMalloc((void**) &d_b_i, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_f, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_c, LSTM_size*1*sizeof(float));
	cudaMalloc((void**) &d_b_o, LSTM_size*1*sizeof(float));
	
	cudaMalloc((void**) &d_M_i, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_M_f, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_M_c, LSTM_size*LSTM_size*sizeof(float));
	cudaMalloc((void**) &d_M_o, LSTM_size*LSTM_size*sizeof(float));

	if(lower_layer.lower_input && upper_layer.source_side) {
		
		cudaMalloc((void**) &d_U_i, LSTM_size*LSTM_size*sizeof(float));
		cudaMalloc((void**) &d_U_f, LSTM_size*LSTM_size*sizeof(float));
		cudaMalloc((void**) &d_U_c, LSTM_size*LSTM_size*sizeof(float));
		cudaMalloc((void**) &d_U_o, LSTM_size*LSTM_size*sizeof(float));
		

		/*	
		float **h_bloew_bi_mat = (float **)malloc(200*sizeof(float*));
		for(int i=0; i<200; i++) {
			cudaMalloc((void**)&h_bloew_bi_mat[i], LSTM_size*sizeof(float));
		}
		cudaMalloc((void**)&d_below_bi_mat, 200*sizeof(float));
		cudaMemcpy(d_below_bi_mat,h_bloew_bi_mat,200*sizeof(float*),cudaMemcpyHostToDevice);
		*/
	}

	cudaMalloc((void**)&d_temp_1, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_2, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_3, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_4, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_5, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_6, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_7, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_temp_8, LSTM_size*minibatch_size*sizeof(float));
	
	if(lower_layer.lower_input && upper_layer.source_side) {
		cudaMalloc((void**)&d_temp_1_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_3_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_5_bi, LSTM_size*minibatch_size*sizeof(float));
		cudaMalloc((void**)&d_temp_7_bi, LSTM_size*minibatch_size*sizeof(float));
	}

	//node
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
	
	cudaMalloc((void**)&d_h_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float));
	cudaMalloc((void**)&d_c_t_prev_tmp, LSTM_size*minibatch_size*sizeof(float));
	
	cudaMalloc((void**)&d_h_t_below, LSTM_size*minibatch_size*sizeof(float));
	
	//
	cudaMalloc((void**)&d_h_t_below_bi, LSTM_size*200*sizeof(float));

	cudaMalloc((void**)&d_father_idx, minibatch_size*sizeof(int));
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Hidden_To_Hidden_Layer::prepare_forward(float *d_h_t_prev, float *d_c_t_prev) {
	
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;
}

void Hidden_To_Hidden_Layer::prepare_forward_decode(int *father_idx,  int B, cudaEvent_t &prev_event) {
	
	cudaMemcpy(d_father_idx, father_idx, B*sizeof(int), cudaMemcpyHostToDevice);
	dim3 block_shape(256,1,1);
	dim3 grid_shape(B,(LSTM_size + block_shape.x - 1)/block_shape.x,1);
	cudaStreamWaitEvent(gpu_info.s0, prev_event, 0);

	lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_h_t_prev_tmp,d_h_t,d_father_idx,LSTM_size);
	lookup_kernel<<<grid_shape,block_shape,0,gpu_info.s0>>>(d_c_t_prev_tmp,d_c_t,d_father_idx,LSTM_size);
	
	this->d_h_t_prev = d_h_t_prev_tmp;
	this->d_c_t_prev = d_c_t_prev_tmp;

	//cout<<"show Hidden_To_Hidden_Layer d_h_t_prev: "<<endl;
	//show_matrix(d_h_t_prev, LSTM_size, B);
	
	//cout<<"show Hidden_To_Hidden_Layer d_c_t_prev: "<<endl;
	//show_matrix(d_c_t_prev, LSTM_size, B);
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();	

}

void Hidden_To_Hidden_Layer::forward_prop_sync(cudaStream_t &my_s) {
	
	if(lower_layer.lower_input) {
		cudaStreamWaitEvent(my_s, lower_layer.input_layer->gpu_info.h_t_below_transfer, 0);
	}
	else {
		cudaStreamWaitEvent(my_s, lower_layer.hidden_layer->gpu_info.h_t_below_transfer, 0);
	}
}

void Hidden_To_Hidden_Layer::forward_prop(int index, int T, int B) {
	
	minibatch_size = B;
	
	bool flag = true;

	float alpha = 1.0;
	float beta = 0.0;

	cudaSetDevice(gpu_num);
	dim3 block_shape1(256,1,1);
	dim3 grid_shape1((LSTM_size + block_shape1.x - 1)/block_shape1.x,minibatch_size,1); // (4,minibatch_size,1)
	
	//d_M_ * d_h_t_below
	cublasSetStream(gpu_info.handle, gpu_info.s0);
	forward_prop_sync(gpu_info.s0);
	//cudaStreamWaitEvent(gpu_info.s9, gpu_info.e9, 0);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_i,LSTM_size,d_h_t_below,LSTM_size,&beta,d_temp_1,LSTM_size); // (1000,1000), (1000, 1), (1000,1)
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		//if(index == 1) {
		//	cout<<"show d_h_t_below_bi: "<<endl;
		//	show_matrix(d_h_t_below_bi+index*LSTM_size,LSTM_size,1);
		
		//	cout<<"show d_U_i: "<<endl;
		//	show_matrix(d_U_i,LSTM_size,LSTM_size);
		//}
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_U_i,LSTM_size,d_h_t_below_bi+index*LSTM_size,LSTM_size,&beta,d_temp_1_bi,LSTM_size);
	}
	cudaEventRecord(gpu_info.e0, gpu_info.s0);
	
	cublasSetStream(gpu_info.handle, gpu_info.s1);
	forward_prop_sync(gpu_info.s1);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_f,LSTM_size,d_h_t_below,LSTM_size,&beta,d_temp_3,LSTM_size); // (H,H), (H,1), (H,1)
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_U_f,LSTM_size,d_h_t_below_bi+index*LSTM_size,LSTM_size,&beta,d_temp_3_bi,LSTM_size); 
	}
	cudaEventRecord(gpu_info.e1, gpu_info.s1);

	cublasSetStream(gpu_info.handle, gpu_info.s2);
	forward_prop_sync(gpu_info.s2);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_c,LSTM_size,d_h_t_below,LSTM_size,&beta,d_temp_5,LSTM_size); 
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_U_c,LSTM_size,d_h_t_below_bi+index*LSTM_size,LSTM_size,&beta,d_temp_5_bi,LSTM_size); 	
	}
	cudaEventRecord(gpu_info.e2, gpu_info.s2);
	
	cublasSetStream(gpu_info.handle, gpu_info.s3);
	forward_prop_sync(gpu_info.s3);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_o,LSTM_size,d_h_t_below,LSTM_size,&beta,d_temp_7,LSTM_size); 
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_U_o,LSTM_size,d_h_t_below_bi+index*LSTM_size,LSTM_size,&beta,d_temp_7_bi,LSTM_size); 
	}
	cudaEventRecord(gpu_info.e3, gpu_info.s3);
	
	if(lower_layer.lower_input && upper_layer.source_side) {
	
	}
	

	// d_W_ * h_{t-1}
	cublasSetStream(gpu_info.handle, gpu_info.s4);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hi,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_2,LSTM_size); // (1000, 1000), (1000, 1), (1000,1)

	cublasSetStream(gpu_info.handle, gpu_info.s5);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hf,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_4,LSTM_size); // (H,H), (H,1), (H,H)

	cublasSetStream(gpu_info.handle, gpu_info.s6);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_hc,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_6,LSTM_size); 

	cublasSetStream(gpu_info.handle, gpu_info.s7);
	cublasSgemm(gpu_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_W_ho,LSTM_size,d_h_t_prev,LSTM_size,&beta,d_temp_8,LSTM_size); 
	

	
	// gate
	cudaStreamWaitEvent(gpu_info.s4, gpu_info.e0, 0);
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		// forward_sigmoid_kernel_ == forward_sigmoid_kernel_feed
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s4>>>(d_i_t, d_temp_1, d_temp_1_bi, d_temp_2, d_b_i, LSTM_size); //d_i_t   (1000,1)

		//cout<<"show d_i_t: "<<endl;
		//show_matrix(d_i_t, LSTM_size, minibatch_size);
	}
	else {
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s4>>>(d_i_t, d_temp_1,d_temp_2, d_b_i, LSTM_size); //d_i_t   (1000,1)
	}
	cudaEventRecord(gpu_info.e4, gpu_info.s4);


	cudaStreamWaitEvent(gpu_info.s5, gpu_info.e1, 0);
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s5>>>(d_f_t, d_temp_3, d_temp_3_bi, d_temp_4, d_b_f, LSTM_size); //d_f_t
	}
	else {
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s5>>>(d_f_t, d_temp_3,d_temp_4, d_b_f, LSTM_size); //d_f_t
	}
	cudaEventRecord(gpu_info.e5, gpu_info.s5);


	cudaStreamWaitEvent(gpu_info.s6, gpu_info.e2, 0);
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		//forward_tanh_kernel_bi == forward_tanh_kernel_feed
		forward_tanh_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s6>>>(d_c_prime_t, d_temp_5, d_temp_5_bi, d_temp_6, d_b_c, LSTM_size); //d_c_prime_t
	}
	else {
		forward_tanh_kernel<<<grid_shape1,block_shape1,0,gpu_info.s6>>>(d_c_prime_t, d_temp_5, d_temp_6, d_b_c, LSTM_size); //d_c_prime_t
	}
	cudaEventRecord(gpu_info.e6, gpu_info.s6);


	cudaStreamWaitEvent(gpu_info.s7, gpu_info.e3, 0);
	if(lower_layer.lower_input && upper_layer.source_side && flag) {
		forward_sigmoid_kernel_feed<<<grid_shape1,block_shape1,0,gpu_info.s7>>>(d_o_t, d_temp_7, d_temp_7_bi, d_temp_8, d_b_o, LSTM_size); //d_o_t
	}
	else {
		forward_sigmoid_kernel<<<grid_shape1,block_shape1,0,gpu_info.s7>>>(d_o_t, d_temp_7, d_temp_8, d_b_o, LSTM_size); //d_o_t
	}
	cudaEventRecord(gpu_info.e7, gpu_info.s7);

	
	// c_t
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e4, 0);
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e5, 0);
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e6, 0);
	forward_c_t_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_c_t, d_f_t, d_c_t_prev, d_i_t, d_c_prime_t, LSTM_size); 

	// h_t
	cudaStreamWaitEvent(gpu_info.s0, gpu_info.e7, 0);
	forward_h_t_kernel<<<grid_shape1,block_shape1,0,gpu_info.s0>>>(d_h_t, d_o_t, d_c_t, LSTM_size); 

	
	if(upper_layer.upper_softmax) {
		if(!upper_layer.source_side) {
			upper_layer.softmax->d_h_t_below = d_h_t;
		}
	}
	else {
		upper_layer.hidden_layer->d_h_t_below = d_h_t;
	}

	cudaEventRecord(gpu_info.h_t_below_transfer, gpu_info.s0);
	

	// highest layer and source side
	if(upper_layer.upper_softmax && upper_layer.source_side) {

		cudaMemcpy(upper_layer.softmax->d_h_t_source+index*LSTM_size, d_h_t, LSTM_size*sizeof(float), cudaMemcpyDeviceToDevice);
	}

}

