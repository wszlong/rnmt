
#ifndef CUSTOM_KERNELS_H
#define CUSTOM_KERNELS_H

#include <thrust/transform_reduce.h>
#include <assert.h>

#define NORM_THREADS 256
#define NUM_ATTENTION_THREADS 128
#define SOFTMAX_THREADS 256

//---------------------Input data formatting kernels--------------------------

//transform vocab indices with -1's and numbers to all 0's and 1's
__global__ 
void vocab_to_01(int *d_vocab_indicies_01,int *d_vocab_indicies,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies_01[i] = 0;
		}
		else {
			d_vocab_indicies_01[i] = 1;
		}
	}
}

//gets rid of all -1's and replaces them with index 0
__global__ 
void vocab_to_nonM1(int *d_vocab_indicies_nonM1,int *d_vocab_indicies,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies_nonM1[i] = 0;
		}
		else {
			d_vocab_indicies_nonM1[i] = d_vocab_indicies[i];
		}
	}
}


__global__
void setup_reverse_indicies(int *d_reverse_unique_indicies,int *d_unique_indicies,int curr_num_unique) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<curr_num_unique; i+=gridDim.x*blockDim.x) {
		//int temp = d_unique_indicies[i];
		//printf("%d\n",d_unique_indicies[i]);
		d_reverse_unique_indicies[d_unique_indicies[i]] = i;
	}
}


//softmax kernel to preprocess data
template<typename dType>
__global__ 
void vocab_softmax(int *d_vocab_indicies,int *d_vocab_indicies_01,dType *d_vocab_indicies_01_float,int total_length) {
	for(int i= threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies[i] = 0;
			d_vocab_indicies_01[i] = 0;
			d_vocab_indicies_01_float[i] = 0;
		}
		else {
			d_vocab_indicies_01[i] = 1;
			d_vocab_indicies_01_float[i] = 1;
		}
	}
}

template<typename dType>
__global__ 
void sparse_lookup_kernel(dType *d_lookup, dType *d_W,int *d_vocab_indices, int minibatch_size,int hiddenstate_size)
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		d_lookup[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_W[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)];
	}
}

__global__
void forward_sigmoid_kernel(float *d_final,float *temp1,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0f/(1.0f + expf(-1.0f*temp_val));
	}
}

__global__
void forward_sigmoid_kernel(double *d_final,double *temp1,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0/(1.0 + exp(-1.0*temp_val));
	}
}


__global__
void forward_sigmoid_kernel_bi(float *d_final,float *temp1,float *temp1_bi,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp1_bi[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0f/(1.0f + expf(-1.0f*temp_val));
	}
}

__global__
void forward_sigmoid_kernel_bi(double *d_final,double *temp1,double *temp1_bi,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp1_bi[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0/(1.0 + exp(-1.0*temp_val));
	}
}


template<typename dType>
__global__
void forward_sigmoid_kernel_feed(dType *d_final,dType *temp1,dType *temp2,dType *temp3,dType *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		dType temp_val = temp1[index] + temp2[index] + temp3[index] +d_bias[idx];
		d_final[index] = 1.0/(1.0 + exp(-1.0*temp_val));
	}
}

__global__
void forward_tanh_kernel(float *d_final,float *temp1,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanhf(temp_val);
	}
}

__global__
void forward_tanh_kernel(double *d_final,double *temp1,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanh(temp_val);
	}
}

__global__
void forward_tanh_kernel_bi(float *d_final,float *temp1,float *temp1_bi,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp1_bi[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanhf(temp_val);
	}
}

__global__
void forward_tanh_kernel_bi(double *d_final,double *temp1,double *temp1_bi,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp1_bi[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanh(temp_val);
	}
}

template<typename dType>
__global__
void forward_tanh_kernel_feed(dType *d_final,dType *temp1,dType *temp2,dType *temp3,dType *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		dType temp_val = temp1[index] + temp2[index] + temp3[index] + d_bias[idx];
		d_final[index] = tanh_wrapper(temp_val);
	}
}

// compute c_t
template<typename dType>
__global__
void forward_c_t_kernel(dType *d_c_t,dType *d_f_t, dType *d_c_t_prev,dType *d_i_t,dType *d_c_prime_t_tanh,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_c_t[index] = d_f_t[index] * d_c_t_prev[index] + d_i_t[index] * d_c_prime_t_tanh[index];
	}
}

//compute h_t
__global__
void forward_h_t_kernel(float *d_h_t,float *d_o_t, float *d_c_t,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_o_t[index] * tanhf(d_c_t[index]);
	}
}

__global__
void forward_h_t_kernel(double *d_h_t,double *d_o_t,double *d_c_t,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_o_t[index] * tanh(d_c_t[index]);
	}
}


template<typename dType>
__global__ 
void zero_c_t_and_h_t(dType *d_h_t,dType *d_c_t,int *d_vocab_indices_01,int hiddenstate_size) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
  	int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_h_t[index] * d_vocab_indices_01[blockIdx.x];
		d_c_t[index] = d_c_t[index] * d_vocab_indices_01[blockIdx.x];
	}
}

//attention_node aligment
template<typename dType>
__global__
void alignment_kernel(dType *d_alignments,dType **d_total_hs_mat,dType *d_h_t,int LSTM_size,int minibatch_size,int *d_batch_info)
{
	    __shared__ dType buffer[NUM_ATTENTION_THREADS];
		const int tid = threadIdx.x;
	    for(int i = blockIdx.x;i < minibatch_size * (d_batch_info[0] + d_batch_info[minibatch_size]);i += gridDim.x){
	    	buffer[tid] = 0;
                int minibatch_index = i % minibatch_size;
                int s_index = i / minibatch_size; 
		    for(int j= threadIdx.x;j < LSTM_size;j+=blockDim.x){
				buffer[tid] += d_total_hs_mat[s_index][IDX2C(j,minibatch_index,LSTM_size)] * d_h_t[IDX2C(j,minibatch_index,LSTM_size)];
			}
		   __syncthreads();
		   for(int stride = NUM_ATTENTION_THREADS/2;stride > 0;stride>>=1){
		       if(tid < stride){
			       buffer[tid] += buffer[stride + tid];
			 }
		  __syncthreads();
		 }
	 __syncthreads();
														    
	dType sum_k = buffer[0];   
    if(tid == 0){
	    d_alignments[IDX2C(minibatch_index,s_index,minibatch_size)] = sum_k;
	}
    __syncthreads();
	}
}


template<typename dType>
__global__
void normalization_alignment_kernel(dType *d_normal_alignments,dType *d_alignments,int minibatch_size,int *d_batch_info)
{
	    int tid = threadIdx.x;
            dType max_val = 0;
            dType sum = 0;
            for (int i = d_batch_info[tid + minibatch_size]; i < (d_batch_info[0] +  d_batch_info[minibatch_size]); i++)
            {
		if(d_alignments[IDX2C(tid,i,minibatch_size)] > max_val)
                {
		    max_val = d_alignments[IDX2C(tid,i,minibatch_size)];
                }
	    }
	    for(int i = d_batch_info[tid + minibatch_size];i < d_batch_info[0] + d_batch_info[minibatch_size];i++)
	    {
		    d_alignments[IDX2C(tid,i,minibatch_size)] = exp(d_alignments[IDX2C(tid,i,minibatch_size)] - max_val);
                    sum += d_alignments[IDX2C(tid,i,minibatch_size)];
        }
	    if (sum!=0)
            {
                   for(int i = d_batch_info[tid + minibatch_size];i < d_batch_info[0] + d_batch_info[minibatch_size];i++)
	           {
		        d_normal_alignments[IDX2C(tid,i,minibatch_size)]=d_alignments[IDX2C(tid,i,minibatch_size)]/sum;
	           }
            }
}

//c_t is lstm_size * minibatch_size
template<typename dType>
__global__
void create_my_c_t_kernel(dType *d_alignments,dType **d_total_hs_mat,dType *d_c_t,int LSTM_size,int minibatch_size, int *d_batch_info)
{
     for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
         d_c_t[i]=0;
         int minibatch_index = i/LSTM_size;
         int lstm_index= i%LSTM_size;		
         for(int j=0; j<d_batch_info[0]+d_batch_info[minibatch_size]; j++) {
             d_c_t[i] +=d_alignments[IDX2C(minibatch_index,j,minibatch_size)] * d_total_hs_mat[j][IDX2C(lstm_index,minibatch_index,LSTM_size)];
         }
     }
}


template<typename dType>
__global__
void tanh_att_forward_kernel(dType *d_output,dType *d_in1,dType *d_in2,dType *d_bias,int LSTM_size,int minibatch_size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_output[i] = tanh_wrapper(d_in1[i] + d_in2[i] + d_bias[i%LSTM_size]);
	}
}

template<typename dType>
__global__
void zero_h_t(dType *d_h_t, int *d_01_mask,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_h_t[i] *= d_01_mask[i/LSTM_size];
	}
}


//softmax 
template<typename dType>
__global__ 
void matrix_bias_kernel(int hiddenstate_size, dType *d_mat,dType *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
	d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] + d_vec[idx];
	}
}

template<typename dType>
__global__
void outputdist_overflow_prevention_kernel(dType *output, dType *input, int dim) {
	__shared__ dType buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	//buffer[threadIdx.x] = -FLT_MAX;
	buffer[threadIdx.x] = -FLT_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		dType z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceMax(buffer,tid);
	// }

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	dType max_k = buffer[0];
	__syncthreads(); //THIS MUST BE HERE
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		dType z = cuda_exp_wrapper(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceSum(buffer,tid);
	// }

  	__syncthreads();

  	// normalize the softmax
	dType sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = output_k[i] / sum_k;
	}
}


template<typename dType>
__global__
void train_perplexity_kernel(int *d_output_vocab_indices_single,int *d_output_vocab_indices_01_single,dType *d_outputdist,
	double *train_perplexity,int minibatch_size,int output_vocab_size) 
{
	for(int i= 0; i<minibatch_size; i++) {
		if(d_output_vocab_indices_01_single[i]==1) {
			train_perplexity[0]+= log( (double) d_outputdist[IDX2C(d_output_vocab_indices_single[i],i,output_vocab_size)]);
		}
	}
}

//*******************************backward kernel************************************//

//*************softmax_node
//This kernel adds a matrices rows to a matrices columns, which ones depend on the index
//hiddenstate_size refers to the number of rows in d_mat_final and also d_mat_col
template<typename dType>
__global__
void matrix_row_to_matrix_column_kernel(dType *d_mat_final,dType *d_mat_col,dType *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)] + \
		d_mat_row[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)];
	}
}

template<typename dType>
__global__ 
void zero_columns_kernel_128(int hiddenstate_size, dType *d_mat,int *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
	d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_vec[blockIdx.x];
	}
}

//This kernel adds a matrices columns to a matrices rows, which ones depend on the index
//hiddenstate_size refers to the number of rows in d_mat_final and also d_mat_col
__global__
void matrix_column_to_matrix_row_kernel(float *d_mat_final,float *d_mat_col,float *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		atomicAdd(&d_mat_final[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)],d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)]);
	}
}

__global__
void matrix_column_to_matrix_row_kernel(double *d_mat_final,double *d_mat_col,double *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		atomicAddDouble(&d_mat_final[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)],d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)]);
	}
}

//add ones to b_d bias unit
__global__
void add_ones_b_d_grad(float *d_b_d_grad,int *d_output_vocab_indices_01,int *d_output_vocab_indices,int minibatch_size) {
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < minibatch_size && d_output_vocab_indices_01[idx]==1) {
		atomicAdd(&d_b_d_grad[d_output_vocab_indices[idx]],1);
	}
}

__global__
void add_ones_b_d_grad(double *d_b_d_grad,int *d_output_vocab_indices_01,int *d_output_vocab_indices,int minibatch_size) {
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < minibatch_size && d_output_vocab_indices_01[idx]==1) {
		atomicAddDouble(&d_b_d_grad[d_output_vocab_indices[idx]],1);
	}
}


//*********************attention_node

template<typename dType>
__global__
void add_two_mats_kernel(dType *d_mat1,dType *d_mat2,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat1[i] = d_mat1[i] + d_mat2[i];
	}
}

//void zero_h_t

template<typename dType>
__global__
void tanh_grad_kernel(dType *d_output,dType *d_input_Error,dType *d_tanh_val,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_output[i] = d_input_Error[i] * (1- d_tanh_val[i]*d_tanh_val[i]);
	}
}	

// errr_to_alignments is minibatch*length
template<typename dType>
__global__
void my_error_alignments_kernel(dType *d_ERRnTOt_as, dType *d_ERRnTOt_ct,dType **d_total_hs_mat,int LSTM_size, int minibatch_size,int *d_batch_info)
{
   __shared__ dType buffer[NUM_ATTENTION_THREADS];
   const int tid = threadIdx.x;
   for(int i = blockIdx.x; i<(d_batch_info[0]+d_batch_info[minibatch_size])*minibatch_size; i+=gridDim.x)
  {
       buffer[tid] = 0;
       int minibatch_index=i%minibatch_size;
       int s_index=i/minibatch_size;
       for(int j=threadIdx.x; j<LSTM_size; j+=blockDim.x) 
       {
           buffer[tid] += d_ERRnTOt_ct[IDX2C(j,minibatch_index,LSTM_size)] * d_total_hs_mat[s_index][IDX2C(j,minibatch_index,LSTM_size)];
       }
       __syncthreads();
       for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) 
       {
           if(tid < stride) 
           {
                 buffer[tid] += buffer[stride + tid];
	    }
           __syncthreads();
       }
       __syncthreads();
       dType sum_k = buffer[0];
       if(tid==0) 
      {
             d_ERRnTOt_as[IDX2C(minibatch_index,s_index,minibatch_size)] = sum_k; 
       }
        __syncthreads();
  }
}

template<typename dType>
__global__
void my_error_hs_kernel(dType *d_ERRnTOt_ct, dType *d_alignments,dType **d_total_hs_error,int LSTM_size,int minibatch_size,int *d_batch_info)
{
      for(int i=blockIdx.x; i < minibatch_size*(d_batch_info[0]+d_batch_info[minibatch_size]); i+=gridDim.x)
      {
          int minibatch_index = i%minibatch_size;
          int s_index = i/minibatch_size;
          for(int j= threadIdx.x; j<LSTM_size; j+=blockDim.x) 
          {
              d_total_hs_error[s_index][IDX2C(j,minibatch_index,LSTM_size)]+= d_ERRnTOt_ct[IDX2C(j,minibatch_index,LSTM_size)]*d_alignments[IDX2C(minibatch_index,s_index,minibatch_size)];
	   }
      }
} 

template<typename dType>
__global__
void my_error_alignments_to_hsht_kernel(dType *d_ERRnTOt_as, dType *d_ERRnTOt_hsht,dType *d_alignments,int minibatch_size,int *d_batch_info)
{
for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<(d_batch_info[0]+d_batch_info[minibatch_size])*minibatch_size; i+=gridDim.x*blockDim.x) {
                int length=d_batch_info[0]+d_batch_info[minibatch_size];
		int minibatch_index = i%minibatch_size;
		int s_index = i/minibatch_size;
		d_ERRnTOt_hsht[i] = d_ERRnTOt_as[IDX2C(minibatch_index,s_index,minibatch_size)] * d_alignments[IDX2C(minibatch_index,s_index,minibatch_size)] * ( 1-d_alignments[IDX2C(minibatch_index,s_index,minibatch_size)]);
		for(int j=0; j<length; j++) {
			if(j!=s_index) {
				d_ERRnTOt_hsht[i] += -1*d_ERRnTOt_as[IDX2C(minibatch_index,j,minibatch_size)] * d_alignments[IDX2C(minibatch_index,j,minibatch_size)] * d_alignments[IDX2C(minibatch_index,s_index,minibatch_size)];
			}
		}
	}


}

template<typename dType>
__global__
void my_error_hsht_to_hs_kernel(dType **d_total_hs_error, dType *d_ERRnTOt_hsht,dType *d_h_t,int LSTM_size,int minibatch_size,int *d_batch_info)
{
    for(int i=blockIdx.x; i < minibatch_size*(d_batch_info[0]+d_batch_info[minibatch_size]); i+=gridDim.x) 
   {
          //int length=d_batch_info[0]+d_batch_info[minibatch_size]; //
          int minibatch_index = i%minibatch_size;
          int s_index = i/minibatch_size;
          for(int j= threadIdx.x; j<LSTM_size; j+=blockDim.x)
          {
              d_total_hs_error[s_index][IDX2C(j,minibatch_index,LSTM_size)]+=d_ERRnTOt_hsht[IDX2C(minibatch_index,s_index,minibatch_size)]*d_h_t[IDX2C(j,minibatch_index,LSTM_size)];
          } 

   }
}

template<typename dType>
__global__
void my_error_hsht_to_ht_kernel(dType *d_ERRnTOt_ht_p1, dType *d_ERRnTOt_hsht,dType **d_total_hs_mat,int LSTM_size,int minibatch_size,int *d_batch_info)
{
    for(int i=blockIdx.x; i < minibatch_size; i+=gridDim.x)
   {
          int length=d_batch_info[0]+d_batch_info[minibatch_size];
          int minibatch_index = i;
          for(int j= threadIdx.x; j<LSTM_size; j+=blockDim.x)
          {
              for(int k=0; k<length; k++)
              d_ERRnTOt_ht_p1[IDX2C(j,minibatch_index,LSTM_size)]+=d_ERRnTOt_hsht[IDX2C(minibatch_index,k,minibatch_size)]*d_total_hs_mat[k][IDX2C(j,minibatch_index,LSTM_size)];
          }

   }
}


//********************LSTM_HH, LSTM

__global__ 
void d_ERRt_ct_kernel(float *d_d_ERRt_ct,float *d_d_ERRnTOt_ht,float *d_o_t,float *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
  		float val = tanhf(d_c_t[index]);
		d_d_ERRt_ct[index] = d_d_ERRnTOt_ht[index] * d_o_t[index] * (1.0f - val*val);
	}
}


__global__ 
void d_ERRt_ct_kernel(double *d_d_ERRt_ct,double *d_d_ERRnTOt_ht,double *d_o_t,double *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
  		double val = tanh(d_c_t[index]);
		d_d_ERRt_ct[index] = d_d_ERRnTOt_ht[index] * d_o_t[index] * (1.0f - val*val);
	}
}

template<typename dType>
__global__ 
void zero_columns_kernel(int hiddenstate_size, dType *d_mat,int *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
		d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_vec[blockIdx.x];
	}
}

__global__ 
void d_ERRnTOt_ot_kernel(float *d_d_ERRnTOt_ot,float *d_d_ERRnTOt_ht,float *d_o_t,float *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_ot[index] = d_d_ERRnTOt_ht[index] *  tanhf(d_c_t[index]) * d_o_t[index] * (1 - d_o_t[index]);
	}
}


__global__ 
void d_ERRnTOt_ot_kernel(double *d_d_ERRnTOt_ot,double *d_d_ERRnTOt_ht,double *d_o_t,double *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_ot[index] = d_d_ERRnTOt_ht[index] *  tanh(d_c_t[index]) * d_o_t[index] * (1 - d_o_t[index]);
	}
}

//For floats or doubles
template<typename dType>
__global__ 
void d_ERRnTOt_ft_it_kernel(dType *d_d_ERRnTOt,dType *d_d_ERRnTOt_ct,dType *d_single_err,dType *d_double_err,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt[index] = d_d_ERRnTOt_ct[index] * d_single_err[index] * d_double_err[index] * (1 - d_double_err[index]);
	}
}

template<typename dType>
__global__ 
void d_ERRnTOt_tanhcpt_kernel(dType *d_d_ERRnTOt_tanhcpt,dType *d_d_ERRnTOt_ct,dType *d_i_t,dType *d_c_prime_t_tanh,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_tanhcpt[index] = d_d_ERRnTOt_ct[index] * d_i_t[index] * (1 -d_c_prime_t_tanh[index]*d_c_prime_t_tanh[index]);
	}
}

template<typename dType>
__global__ 
void add_four_matrices_kernel(dType *d_final,dType *d_mat1,dType *d_mat2,dType *d_mat3,dType *d_mat4,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_final[index] = d_mat1[index] + d_mat2[index] + d_mat3[index] + d_mat4[index];
	}
}

template<typename dType>
__global__ 
void elementwise_mult_kernel(dType *d_mat1,dType *d_mat2,dType *d_final,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_mat1[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_mat2[IDX2C(idx,blockIdx.x,hiddenstate_size)];
	}
}

template<typename dType> 
__global__
void W_small_gradient_kernel(dType *d_small_W_grad,int *d_reverse_unique_indicies,dType *temp1,dType *temp2,dType *temp3,
	dType *temp4,int *d_vocab_indicies,int LSTM_size,int minibatch_size) 
{	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {
		int vocab_index = d_vocab_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			dType sum = temp1[IDX2C(i,k,LSTM_size)] + temp2[IDX2C(i,k,LSTM_size)] + temp3[IDX2C(i,k,LSTM_size)] + temp4[IDX2C(i,k,LSTM_size)];
			atomicAdd(&(d_small_W_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),sum);
		}
	}
}


//*************update model


struct scale_functor {
	const int minibatch_size;

	scale_functor(int _minibatch_size) : minibatch_size(_minibatch_size) {}

	__host__ __device__ void operator()(float &x) {
		x = (1.0f/minibatch_size)*x;
	}
	__host__ __device__ void operator()(double &x) {
		x = (1.0/minibatch_size)*x;
	}
};

template<typename dType>
struct re_scale_norm_functor {
	const dType norm_threshold;
	const dType norm;

	re_scale_norm_functor(dType _norm_threshold,dType _norm) : norm_threshold(_norm_threshold),norm(_norm) {}

	__host__ __device__ void operator()(dType &x) {
		x = (norm_threshold/norm)*x;
	}
};


template<typename dType>
__global__
void add_grad_vecs(dType *vec1,dType *vec2,dType learning_rate,int size) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < size) {
		vec1[idx] = learning_rate*vec2[idx] + vec1[idx];
	}
}

template<typename dType>
__global__
void update_sparse_grad(dType *d_mat,dType *d_small_grad,int *d_unique_indicies,int curr_num_unique,dType learning_rate,int LSTM_size) {
	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < curr_num_unique; k+=gridDim.x) {
		int vocab_index = d_unique_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			//atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),d_temp_D_grad[IDX2C(i,k,LSTM_size)]);
			d_mat[IDX2C(i,vocab_index,LSTM_size)] += learning_rate*d_small_grad[IDX2C(i,k,LSTM_size)];
		}
	}
}

template<typename dType>
__global__
void gradient_update_mats(dType *d_mat,dType *d_mat_grad,dType learning_rate,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat[i]+= learning_rate * d_mat_grad[i];
	}
}


//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
template<typename dType>
__device__ 
void warpReduceSum(volatile dType* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename dType>
__device__ 
void warpReduceMax(volatile dType* sdata, int tid) {
	sdata[tid] = (sdata[tid] > sdata[32 + tid]) ? sdata[tid] : sdata[32 + tid];
	sdata[tid] = (sdata[tid] > sdata[16 + tid]) ? sdata[tid] : sdata[16 + tid];
	sdata[tid] = (sdata[tid] > sdata[8 + tid]) ? sdata[tid] : sdata[8 + tid];
	sdata[tid] = (sdata[tid] > sdata[4 + tid]) ? sdata[tid] : sdata[4 + tid];
	sdata[tid] = (sdata[tid] > sdata[2 + tid]) ? sdata[tid] : sdata[2 + tid];
	sdata[tid] = (sdata[tid] > sdata[1 + tid]) ? sdata[tid] : sdata[1 + tid];
}


template<typename dType,typename dType2>
__global__
void outputdist_perplexity_kernel(dType2 *output, dType *input, int dim,bool print_partition_function,double *d_partition_vals) {
	__shared__ double buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType2 *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	buffer[threadIdx.x] = -DBL_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		double z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=SOFTMAX_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceMax(buffer,tid);
	}

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	double max_k = buffer[0];
	__syncthreads();
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		double z = cuda_exp_wrapper(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=SOFTMAX_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}

  	__syncthreads();

  	// normalize the softmax
	double sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = cuda_log_wrapper(output_k[i]) - cuda_log_wrapper(sum_k);
	}

	if(print_partition_function && threadIdx.x == 0) {
		d_partition_vals[blockIdx.x] = sum_k;
	}
}


template<typename dType>
__global__
void basic_compute_norm_p1(dType *d_gradient,int size,dType *result) {
	__shared__ dType buffer[NORM_THREADS];
	int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
	int i_end = size; //end at dim
	int i_step = blockDim.x*gridDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;


	buffer[tid] = 0;
	for(int i= i_start; i<i_end; i+=i_step) {
		buffer[tid]+=(d_gradient[i]*d_gradient[i]);
	}
	__syncthreads();

	for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}
	__syncthreads();

	if(tid==0) {
		result[blockIdx.x]=buffer[0];
	}
}


template<typename dType>
__global__
void basic_compute_norm_p2(dType *temp_result,dType *final_result) {
	__shared__ dType buffer[NORM_THREADS];

	int tid = threadIdx.x;
	buffer[tid] = temp_result[tid];
	__syncthreads();

	for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}
	__syncthreads();

	if(tid==0) {
		final_result[0]=buffer[0];
	}
}


//clip the norm if it is greater than the threshold
template<typename dType>
void norm_clip_GPU_v2(thrust::device_ptr<dType> &thrust_d_gradient,dType *d_gradient,dType norm_threshold,int size,dType *d_temp_result,dType *d_result) {

	dType norm;
	basic_compute_norm_p1<<<NORM_THREADS,NORM_THREADS>>>(d_gradient,size,d_temp_result);
	basic_compute_norm_p2<<<1,NORM_THREADS>>>(d_temp_result,d_result);
	cudaMemcpy(&norm,d_result,1*sizeof(dType),cudaMemcpyDeviceToHost);
	MY_CUDA::recent_sum = norm;
	norm = std::sqrt(norm);
	if(norm > norm_threshold) {
		//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
		re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
		thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	}
}

//for dropout
template<typename dType>
__global__
void dropout_kernel(dType *d_dropout_mask,dType rate,dType *d_final, int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_final[i] = (d_dropout_mask[i] < rate) * (1/rate) * d_final[i];
	}
}

template<typename dType>
__global__
void W_small_dropout_gradient_kernel(dType *d_small_W_grad,int *d_reverse_unique_indicies,dType *temp1,dType *temp2,dType *temp3,
	dType *temp4,int *d_vocab_indicies,int LSTM_size,int minibatch_size,dType *d_dropout_mask,dType rate) 
{	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {
		int vocab_index = d_vocab_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			dType sum = temp1[IDX2C(i,k,LSTM_size)] + temp2[IDX2C(i,k,LSTM_size)] + temp3[IDX2C(i,k,LSTM_size)] + temp4[IDX2C(i,k,LSTM_size)];
			sum = sum*(rate > d_dropout_mask[IDX2C(i,k,LSTM_size)])*(1/rate);
			atomicAdd(&(d_small_W_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),sum);
		}
	}
}

//adam
template<typename dType>
__global__ 
void update_params_adam(dType *d_W_hi, dType *d_W_hi_grad, dType *d_W_hi_mt, dType *d_W_hi_vt, dType alpha, dType beta_1, dType beta_2, dType epsilon, int time_inex, int hiddenstate_size, int LSTM_size)
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;

	for(int k=blockIdx.x; k<LSTM_size; k+=gridDim.x) {
		if(idx < hiddenstate_size) {
			d_W_hi_mt[IDX2C(idx,k,hiddenstate_size)] = beta_1 * d_W_hi_mt[IDX2C(idx,k,hiddenstate_size)] + (1-beta_1) * d_W_hi_grad[IDX2C(idx,k,hiddenstate_size)];
			d_W_hi_vt[IDX2C(idx,k,hiddenstate_size)] = beta_2 * d_W_hi_vt[IDX2C(idx,k,hiddenstate_size)] + (1-beta_2) * d_W_hi_grad[IDX2C(idx,k,hiddenstate_size)] * d_W_hi_grad[IDX2C(idx,k,hiddenstate_size)];
			d_W_hi[IDX2C(idx,k,hiddenstate_size)] += alpha* (d_W_hi_mt[IDX2C(idx,k,hiddenstate_size)]/(1-pow(beta_1, time_inex))) / (sqrt(d_W_hi_vt[IDX2C(idx,k,hiddenstate_size)] / (1-pow(beta_2,time_inex))) + epsilon);

		}
	}
}

template<typename dType>
__global__
void add_grad_vecs_adam(dType *vec1,dType *vec2,dType *vec1_mt,dType *vec1_vt,dType alpha,dType beta_1,dType beta_2,dType epsilon,int time_inex,int size) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < size) {
		vec1_mt[idx] = beta_1 * vec1_mt[idx] + (1-beta_1) * vec2[idx];
		vec1_vt[idx] = beta_2 * vec1_vt[idx] + (1-beta_2) * vec2[idx] * vec2[idx];
		
		vec1[idx] += alpha * (vec1_mt[idx]/(1-pow(beta_1, time_inex))) / (sqrt(vec1_vt[idx] / (1-pow(beta_2,time_inex))) + epsilon);
		//vec1[idx] = learning_rate*vec2[idx] + vec1[idx];
	}
}

template<typename dType>
__global__
void update_sparse_grad_adam(dType *d_mat,dType *d_small_grad,dType *d_mat_mt,dType *d_mat_vt,dType alpha,dType beta_1,dType beta_2,dType epsilon,int time_inex,int *d_unique_indicies,int curr_num_unique,int LSTM_size) {
	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < curr_num_unique; k+=gridDim.x) {
		int vocab_index = d_unique_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			d_mat_mt[IDX2C(i,vocab_index,LSTM_size)] = beta_1 * d_mat_mt[IDX2C(i,vocab_index,LSTM_size)] + (1-beta_1) * d_small_grad[IDX2C(i,k,LSTM_size)];	
			d_mat_vt[IDX2C(i,vocab_index,LSTM_size)] = beta_2 * d_mat_vt[IDX2C(i,vocab_index,LSTM_size)] + (1-beta_2) * d_small_grad[IDX2C(i,k,LSTM_size)] * d_small_grad[IDX2C(i,k,LSTM_size)];

			d_mat[IDX2C(i,vocab_index,LSTM_size)] += alpha * (d_mat_mt[IDX2C(i,vocab_index,LSTM_size)]/(1-pow(beta_1, time_inex))) / (sqrt(d_mat_vt[IDX2C(i,vocab_index,LSTM_size)] / (1-pow(beta_2,time_inex))) + epsilon);

			//d_mat[IDX2C(i,vocab_index,LSTM_size)] += learning_rate*d_small_grad[IDX2C(i,k,LSTM_size)];
		}
	}
}

template<typename dType>
__global__
void gradient_update_mats_adam(dType *d_mat,dType *d_mat_grad,dType *d_mat_mt,dType *d_mat_vt,dType alpha,dType beta_1,dType beta_2,dType epsilon,int time_index,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat_mt[i] = beta_1 * d_mat_mt[i] + (1-beta_1) * d_mat_grad[i];
		d_mat_vt[i] = beta_2 * d_mat_vt[i] + (1-beta_2) * d_mat_grad[i] * d_mat_grad[i];
		
		d_mat[i] += alpha * (d_mat_mt[i]/(1-pow(beta_1, time_index))) / (sqrt(d_mat_vt[i] / (1-pow(beta_2,time_index))) + epsilon);

		//d_mat[i]+= learning_rate * d_mat_grad[i];
	}
}



#endif
