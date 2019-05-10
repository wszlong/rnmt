
#include <fstream>
#include "transfer_layer.h"

using namespace std;


class Input_To_Hidden_Layer {
	public:

		gpu_info_struct gpu_info;
		int gpu_num;
		
		bool bi_dir;	
		int Embedding_size;
		int LSTM_size;
		int minibatch_size;
		int vocab_size;
		bool feed_input = false;
		
		float *d_W_hi;
		float *d_W_hf;
		float *d_W_hc;
		float *d_W_ho;
		
		float *d_b_i;
		float *d_b_f;
		float *d_b_c;
		float *d_b_o;

		float *d_W;

		float *d_M_i;
		float *d_M_f;
		float *d_M_o;
		float *d_M_c;

		float *d_Q_i;
		float *d_Q_f;
		float *d_Q_o;
		float *d_Q_c;
		

		float *d_temp_1;
		float *d_temp_2;
		float *d_temp_3;
		float *d_temp_4;
		float *d_temp_5;
		float *d_temp_6;
		float *d_temp_7;
		float *d_temp_8;

		float *d_temp_1_feed;
		float *d_temp_2_feed;
		float *d_temp_3_feed;
		float *d_temp_4_feed;

		//node
		int *input_wids;
		int *d_wid;
		float *d_x_t;

		float *d_i_t;
		float *d_f_t;
		float *d_c_prime_t;
		float *d_o_t;

		float *d_c_t;
		float *d_h_t;
		
		float *d_init_hidden_vector;
		float *d_init_cell_vector;
		float *d_h_t_prev;
		float *d_c_t_prev;

		// for feedinput
		float *d_h_t_feed;

		float *d_h_t_prev_tmp; // for prepare_forward_decode
		float *d_c_t_prev_tmp; //
		
		int *d_father_idx;	
	
		upper_transfer_layer upper_layer;		

		Input_To_Hidden_Layer() {};
		
		void init_Input_To_Hidden_Layer(int Embedding_size, int LSTM_size, int minibatch_size, int vocab_size, bool feed_input, int gpu_num, bool bi_dir);

		void init_params();
		void init_param_feed_input();
		void load_weight(ifstream &input);
		void load_weight_feed_input(ifstream &input);

		void prepare_forward(int *input_wids, float * d_h_t_prev, float * d_c_t_prev);
		void forward_prop(int index, int T, int B);

		void prepare_forward_decode(int *tgt_wids, int *father_idx, int B, float *d_final_temp_2, cudaEvent_t &prev_event);

};
