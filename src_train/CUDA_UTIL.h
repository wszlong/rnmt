
#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdlib.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <cmath>
#include <stdio.h>   
#include <stdlib.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <curand.h>
#include "cuda_profiler_api.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

//#include "global_params.h"

//This is used since all cuBLAS storage is column major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


//namespace to hold constants
namespace MY_CUDA {

	bool cont_train = false;
	bool shuffle_data = true;

	std::vector<int> gpu_indicies;

	unsigned int curr_seed = 0;
	boost::random::mt19937 gen;
	double lower = -0.08;
	double upper = 0.08;

	//for stats on gradient norms
	double recent_sum = 0;
	
	//for dump after n epoch
	bool dump_after_nEpoch = false;
	float begin_dump_num = 1.0;
	float curr_dump_num;

}


void devSynchAll() {
 	
	int origin_device;
 	cudaGetDevice(&origin_device);
 	
	//int num_devices;
 	//cudaGetDeviceCount(&num_devices);
 	//for(int i=0; i<num_devices; i++) {
 	//	cudaSetDevice(i);
 	//	cudaDeviceSynchronize();
	//}
	
	for(int i=0; i<MY_CUDA::gpu_indicies.size(); i++) {
 		cudaSetDevice(MY_CUDA::gpu_indicies[i]);
 		cudaDeviceSynchronize();	
	}
 	
	cudaSetDevice(origin_device);
}

void show_matrix(float *d_m, int r, int c)
{
	std::vector<float> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=0;i<min(r,10);i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,5);j++)
        {
			std::cout<<m[IDX2C(i,j,r)]<<' ';
        }
		std::cout<<std::endl;
    }
}

void show_matrix(int *d_m, int r, int c)
{
	std::vector<int> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=0;i<min(r,10);i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,5);j++)
        {
			std::cout<<m[IDX2C(i,j,r)]<<' ';
        }
		std::cout<<std::endl;
    }
}


void show_matrix(double *d_m, int r, int c)
{
	std::vector<double> m;
    m.resize(r*c);
    cudaMemcpy(&m[0], d_m, r*c*sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i=0;i<r;i++)
    for (int i=0;i<min(r,10);i++)
    {
        //for (int j=0;j<c;j++)
        for (int j=0;j<min(c,5);j++)
        {
			std::cout<<m[IDX2C(i,j,r)]<<' ';
        }
		std::cout<<std::endl;
    }
}


void printIntroMessage(global_params &params) {

	if(params.train) {
		cout << "\n\n------------------------Train Info------------------------\n";
		cout << "Minibatch Size: " << params.minibatch_size << "\n";
		cout << "Number of Epochs: " << params.num_epochs << "\n";
		cout << "Learning Rate: " << params.learning_rate << "\n";
		cout << "adam algorithm: " << params.ADAM << "\n";
		if(params.clip_gradient) {
			cout << "Gradient Clipping Threshold per matrix (Norm Ball): " << params.norm_clip << "\n";
		}
	}
	
	cout << "------------------------Model Info------------------------\n";
	
	cout << "Source Vocab Size: " << params.source_vocab_size << "\n";
	cout << "Target Vocab Size: " << params.target_vocab_size << "\n";
	cout << "Number of Hidden Units: " << params.LSTM_size << "\n";
	cout << "Number of Layers: " << params.num_layers << "\n";
	if(params.attent_params.attention_model) {
		cout << "Attention model set as true\n";
		if(params.attent_params.feed_input) {
			cout << "Feed Input set as true\n";
		}
	}

	cout << "---------------------------------------------------------------\n\n";
}

void CUDA_ERROR_WRAPPER(cudaError_t cudaStat,std::string error_message) {

	if ( cudaSuccess != cudaStat ) {
		cout << "Error\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(cudaStat));
		cout << error_message << "\n";
		exit (EXIT_FAILURE);
	}
}

string cublasErrorString(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void CUBLAS_ERROR_WRAPPER(cublasStatus_t cudaStat,std::string error_message) {
      if (cudaStat != CUBLAS_STATUS_SUCCESS) {
		string msg = cublasErrorString(cudaStat);

		cout << error_message << std::endl;
		cout << msg << "\n";

		exit (EXIT_FAILURE);
	}
}

void CUDA_GET_LAST_ERROR(std::string msg) {
	cudaError_t code = cudaGetLastError();
	if ( cudaSuccess != code ) {
		cout << "Error in kernel\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
		cout << msg << "\n";
		exit (EXIT_FAILURE);
	}
}

template<typename dType>
void allocate_Matrix_CPU(dType **h_matrix,int rows,int cols) {
	*h_matrix = (dType *)malloc(rows*cols*sizeof(dType));
}

template<typename dType>
void allocate_Matrix_GPU(dType **d_matrix,int rows,int cols) {
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)d_matrix, rows*cols*sizeof(dType)),"GPU memory allocation failed\n");
}

//Can be used for either double or float, use floats for performance, but doubles for gradient checking
template<typename dType>
void initialize_Matrix(dType *h_matrix,int rows,int cols) {
	boost::uniform_real<> distribution(MY_CUDA::lower,MY_CUDA::upper);
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  (dType)distribution(MY_CUDA::gen);
		}
	}
}

template<typename dType>
void initialize_Matrix_ones(dType *h_matrix,int rows,int cols) {
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  1;
		}
	}
}

template<typename dType>
void initialize_Matrix_zeros(dType *h_matrix,int rows,int cols) {
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  0;
		}
	}
}


template<typename dType>
void set_matrix_cuBLAS(dType *h_matrix,dType *d_matrix,int rows,int cols) {
	CUBLAS_ERROR_WRAPPER(cublasSetMatrix(rows, cols, sizeof(dType), h_matrix, rows, d_matrix, rows),"cuBLAS set matrix failed\n");
}

template<typename dType>
void set_vector_cuBLAS(dType *h_vector,dType *d_vector,int rows) {
	CUBLAS_ERROR_WRAPPER(cublasSetVector(rows, sizeof(dType), h_vector, 1, d_vector, 1),"cuBLAS set vector failed\n");
}

template<typename dType>
void full_matrix_setup(dType **h_matrix,dType **d_matrix,int rows,int cols) {
	allocate_Matrix_CPU(h_matrix,rows,cols);
	initialize_Matrix(*h_matrix,rows,cols);
	allocate_Matrix_GPU(d_matrix,rows,cols);
	set_matrix_cuBLAS(*h_matrix,*d_matrix,rows,cols);

	free(*h_matrix);
}

template<typename dType>
void full_vector_setup_ones(dType **h_vector,dType **d_vector,int rows) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_Matrix_ones(*h_vector,rows,1);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);

	free(*h_vector);
}

template<typename dType>
void full_matrix_setup_0(dType **h_matrix,dType **d_matrix,int rows,int cols) {
	allocate_Matrix_CPU(h_matrix,rows,cols);
	initialize_Matrix_zeros(*h_matrix,rows,cols);
	allocate_Matrix_GPU(d_matrix,rows,cols);
	set_matrix_cuBLAS(*h_matrix,*d_matrix,rows,cols);

	free(*h_matrix);
}

template<typename dType>
void full_vector_setup(dType **h_vector,dType **d_vector,int rows) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_Matrix(*h_vector,rows,1);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);

	free(*h_vector);
}


__device__
inline double tanh_wrapper(double x) {
	return tanh(x);
}


__device__
inline float tanh_wrapper(float x) {
	return tanhf(x);
}


__device__
inline float cuda_exp_wrapper(float x) {
	return expf(x);
}

__device__
inline double cuda_exp_wrapper(double x) {
	return exp(x);
}

__device__
inline float cuda_log_wrapper(float x) {
	return logf(x);
}

__device__
inline double cuda_log_wrapper(double x) {
	return log(x);
}


inline cublasStatus_t cublas_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const float *alpha, const float *A, int lda, 
	 const float *B, int ldb, const float *beta, float *C, int ldc) 
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublas_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const double *alpha, const double *A, int lda, 
	 const double *B, int ldb, const double *beta, double *C, int ldc) 
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


inline cublasStatus_t cublas_gemv_wrapper(cublasHandle_t handle, cublasOperation_t trans, int m, 
	int n, const float *alpha, const float *A, int lda, const float *x, int incx, 
	const float *beta, float *y, int incy) 
{
	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t cublas_gemv_wrapper(cublasHandle_t handle, cublasOperation_t trans, int m, 
	int n, const double *alpha, const double *A, int lda, const double *x, int incx, 
	const double *beta, double *y, int incy) 
{
	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}


inline cublasStatus_t cublas_geam_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	int m, int n, const float *alpha, const float *A, int lda, const float *beta, 
	const float *B, int ldb, float *C, int ldc) 
{
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

}

inline cublasStatus_t cublas_geam_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	int m, int n, const double *alpha, const double *A, int lda, const double *beta, 
	const double *B, int ldb, double *C, int ldc) 
{
	return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

}


//atomic add for doubles,since undefined in cuda
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

//atomic add for doubles,since undefined in cuda
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


void curandGenerateUniform_wrapper(float *d_mask,int size,curandGenerator_t &generator) {
	curandGenerateUniform(generator,d_mask,size);
}

void curandGenerateUniform_wrapper(double *d_mask,int size,curandGenerator_t &generator) {
	curandGenerateUniformDouble(generator,d_mask,size);
}


//note there are no thrust matrices only vectors
template<typename dType>
void initialize_thrust_vector(thrust::host_vector<dType> &h_vec,int size) {
	boost::uniform_real<> distribution(MY_CUDA::lower,MY_CUDA::upper);
	for(int i=0; i<size; i++) {
		h_vec[i] = (dType)distribution(MY_CUDA::gen);
	}
}


#endif
