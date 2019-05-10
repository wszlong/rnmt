
#include <fstream>
#include "transfer_layer.h"

using namespace std;


class Hidden_To_Hidden_Layer {
	public:

		gpu_info_struct gpu_info;
		int gpu_num;

		int LSTM_size;
		int minibatch_size;
		
		float *d_W_hi;
		float *d_W_hf;
		float *d_W_hc;
		float *d_W_ho;
		
		float *d_b_i;
		float *d_b_f;
		float *d_b_c;
		float *d_b_o;

		float *d_M_i;
		float *d_M_f;
		float *d_M_o;
		float *d_M_c;
		
		float *d_U_i;
		float *d_U_f;
		float *d_U_o;
		float *d_U_c;

		float *d_temp_1;
		float *d_temp_2;
		float *d_temp_3;
		float *d_temp_4;
		float *d_temp_5;
		float *d_temp_6;
		float *d_temp_7;
		float *d_temp_8;
		
		float *d_temp_1_bi;
		float *d_temp_3_bi;
		float *d_temp_5_bi;
		float *d_temp_7_bi;

		//node
		float *d_i_t;
		float *d_f_t;
		float *d_c_prime_t;
		float *d_o_t;

		float *d_init_hidden_vector;
		float *d_init_cell_vector;
		
		float *d_h_t;
		float *d_c_t;	
		float *d_h_t_prev;
		float *d_c_t_prev;
		float *d_h_t_below;
		float *d_h_t_below_bi;

		
		float *d_h_t_prev_tmp;
		float *d_c_t_prev_tmp;

		int *d_father_idx;

		
		upper_transfer_layer upper_layer;
		lower_transfer_layer lower_layer;
		
		Hidden_To_Hidden_Layer() {};

		void init_Hidden_To_Hidden_Layer(int LSTM_size, int minibatch_size, int gpu_num);

		void init_params();
		void load_weight(ifstream &input);
		
		void prepare_forward(float * d_h_t_prev, float * d_c_t_prev);
		void forward_prop_sync(cudaStream_t &my_s);
		void forward_prop(int index, int T, int B);

		void prepare_forward_decode(int *father_idx, int B, cudaEvent_t &prev_event);

};
