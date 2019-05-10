
#ifndef DECODER_H
#define DECODER_H

//#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

class Input_To_Hidden_Layer;

class Hidden_To_Hidden_Layer;

class Attention_Softmax_Layer;

class Decoder {

    public:
        map<string,int> src_w2i;
        map<int,string> tgt_i2w;
		vector<int> father_idx;

        string a; //for test
		int B, K, Tmax, T;
		
		bool feed_input = false;

		int Embedding_size, LSTM_size, num_layers, target_vocab_size, source_vocab_size;
        
		int gpu_num, beam_size;
		string input_weight_file;
		string input_vocab_file;
		
		Decoder(string input_weight_file, string input_vocab_file, int beam_size, int gpu_num);
        string translate(string input_sen);
		

		Input_To_Hidden_Layer input_layer_source;
		Input_To_Hidden_Layer input_layer_source_bi;
		Input_To_Hidden_Layer input_layer_target;

		vector<Hidden_To_Hidden_Layer> source_hidden_layers;
		vector<Hidden_To_Hidden_Layer> target_hidden_layers;

		Attention_Softmax_Layer attention_softmax_target;
    
	//private:
		void load_and_init_model(map<string,int> &src_w2i,map<int,string> &tgt_i2w);
		void init_model();
		void init_model_structure();

        vector<int> w2id(string input_sen);
        string id2w(vector<int> output_wids);
        void encode(vector<int> input_wids);
        vector<int> decode();
        void get_next_prob(vector<int> tgt_wids, int index);
        void generate_new_samples(vector<vector<int> > &hyp_samples,vector<float> &hyp_scores,
                vector<vector<int> > &final_samples, vector<float> &final_scores, int &dead_k, vector<int> &tgt_wids, vector<int> &father_idx);

		
		/***
        //encoder
        int *d_wid;
		float *d_x_t;
		//float *d_temp_1, *d_temp_2, *d_temp_3, *d_temp_4, *d_temp_5, *d_temp_6, *d_temp_7, *d_temp_8;
		float *d_h_all, *d_c_all;
		float *d_i_t, *d_f_t, *d_c_prime_t, *d_o_t;

		//encoder second layer
		float *d_temp2_1, *d_temp2_2, *d_temp2_3, *d_temp2_4, *d_temp2_5, *d_temp2_6, *d_temp2_7, *d_temp2_8;
		float *d_h2_all, *d_c2_all;
		float *d_i2_t, *d_f2_t, *d_c2_prime_t, *d_o2_t;
		
		//decode
		int *d_tgt_wids;
		float *d_y_tm1;
		float *d_h_tild_t;

		//float *d_temp_1_feed, *d_temp_2_feed, *d_temp_3_feed, *d_temp_4_feed;
		float *d_h_t_prev, *d_c_t_prev;
		float *d_h2_t_prev, *d_c2_t_prev;
		float *d_h_t, *d_c_t;
		float *d_h2_t, *d_c2_t;
		
		//attention
		float *d_alignment, *d_normal_alignment;
		float *d_c_att_t;
		float *d_final_temp_1, *d_final_temp_2;

		//softmax
		float *d_outputdist, *d_logit_softmax;

		float *d_ones, *d_outputdist_sum;

		//expand hypothesis
		int *d_father_idx;

        //cublasHandle_t handle;
        float alpha;  
        float beta;  
		****/
};

#endif
