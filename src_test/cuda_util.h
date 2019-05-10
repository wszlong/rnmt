
#include <cublas_v2.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void lookup_kernel(float *d_lookup, float *d_W, int *d_wids, int LSTM_size)
{
    int i = threadIdx.x + blockIdx.y*blockDim.x;
    int j = blockIdx.x;
    if(i < LSTM_size) {
        d_lookup[IDX2C(i,j,LSTM_size)] = d_W[IDX2C(i,d_wids[j],LSTM_size)];
	}
}

__forceinline__ __device__ float sigmoidf(float in)
{
	return 1.f / (1.f + expf(-in)); 
}

__global__ void forward_sigmoid_kernel(float *d_final,float *temp1, float *temp2, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx< LSTM_size) {
	   
	   d_final[IDX2C(idx,j,LSTM_size)] = sigmoidf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + bias[idx]);
	   //d_final[idx] = sigmoidf(temp1[idx] + temp2[idx] + bias[idx]);
   }
}

__global__ void forward_sigmoid_kernel_feed(float *d_final,float *temp1, float *temp2, float *temp3, float *bias, int LSTM_size)
{	
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx< LSTM_size) {
	   d_final[IDX2C(idx,j,LSTM_size)] = sigmoidf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + temp3[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_tanh_kernel(float *d_final,float *temp1, float *temp2, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx < LSTM_size) {
   
	   d_final[IDX2C(idx,j,LSTM_size)] = tanhf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_tanh_kernel_feed(float *d_final,float *temp1, float *temp2, float * temp3, float *bias, int LSTM_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y;
   if (idx < LSTM_size) {
   
	   d_final[IDX2C(idx,j,LSTM_size)] = tanhf(temp1[IDX2C(idx,j,LSTM_size)] + temp2[IDX2C(idx,j,LSTM_size)] + temp3[IDX2C(idx,j,LSTM_size)] + bias[idx]);
   }
}

__global__ void forward_c_t_kernel(float *d_c_t, float *d_f_t, float *d_c_t_prev, float *d_i_t, float *d_c_prime_t_tanh, int LSTM_size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < LSTM_size) {
		
		d_c_t[IDX2C(idx,j,LSTM_size)] = d_f_t[IDX2C(idx,j,LSTM_size)] * d_c_t_prev[IDX2C(idx,j,LSTM_size)] + d_i_t[IDX2C(idx,j,LSTM_size)] * d_c_prime_t_tanh[IDX2C(idx,j,LSTM_size)];
		
		//d_c_t[idx] = d_f_t[idx] * d_c_t_prev[idx] + d_i_t[idx] * d_c_prime_t_tanh[idx];
		//d_c_t_store[idx] = d_c_t[idx];
	}
}

__global__ void forward_h_t_kernel(float *d_h_t, float *d_o_t, float *d_c_t, int LSTM_size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < LSTM_size) {
		
		d_h_t[IDX2C(idx,j,LSTM_size)] = d_o_t[IDX2C(idx,j,LSTM_size)] * tanhf(d_c_t[IDX2C(idx,j,LSTM_size)]); //tanh
		
		//d_h_t[idx] = d_o_t[idx] * tanhf(d_c_t[idx]);
		//d_h_t_store[idx] = d_h_t[idx];
	}
}

__global__ void normalization_alignment_kernel(float *d_normal_alignment, float *d_alignment, int minibatch_size, int T) {

	int tid = threadIdx.x;
	if (tid<minibatch_size) {
		float max_val = 0;
		float sum = 0;

		for (int i=0; i<T; i++) {
			if(d_alignment[IDX2C(tid,i,minibatch_size)] > max_val) {
				max_val = d_alignment[IDX2C(tid,i,minibatch_size)];
			}
		}

		for (int i=0; i<T; i++) {
			d_alignment[IDX2C(tid,i,minibatch_size)] = exp(d_alignment[IDX2C(tid,i,minibatch_size)] - max_val); // exp --> double
			sum += d_alignment[IDX2C(tid,i,minibatch_size)];
		}
		
		if (sum != 0) {
			for (int i=0; i<T; i++) {
				d_normal_alignment[IDX2C(tid,i,minibatch_size)] = d_alignment[IDX2C(tid,i,minibatch_size)]/sum;
			}
		}
	}
}

__global__ void tanh_att_forward_kernel(float *d_output, float *d_in1, float *d_in2, float *d_bias, int LSTM_size, int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_output [i] = tanhf(d_in1[i] + d_in2[i] + d_bias[i%LSTM_size]);
	}
}

__global__ void matrix_bias_kernel(float *d_mat, float *d_vec, float *d_mat_final, int vocab_size) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(idx < vocab_size) {
		d_mat_final[IDX2C(idx,j,vocab_size)] = d_mat[IDX2C(idx,j,vocab_size)] + d_vec[idx];
	}
}

__global__ void exp_overflow_prevention(float *m, int rows){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(i<rows){
		//m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]-15); // for prevention overflow
		m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]-10); // for prevention overflow
		//m[IDX2C(i,j,rows)] = expf(m[IDX2C(i,j,rows)]);
	}
}

__global__ void divide(float *v1, float *v2, float *v3, int rows){
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = blockIdx.y;
	if(i<rows){
		v1[IDX2C(i,j,rows)] = v2[IDX2C(i,j,rows)]/v3[j];
	}	
}

__global__ void outputdist_overflow_prevention_kernel(float *output, float *input, int dim) {
	
	__shared__ float buffer[256]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	float *input_k = input + k*dim; //all threads in block start from same index
	float *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	buffer[threadIdx.x] = -FLT_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		float z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=256/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	float max_k = buffer[0];
	__syncthreads();
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		float z = expf(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=256/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}


  	__syncthreads();

  	// normalize the softmax
	float sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = output[i] / sum_k;
	}
}

