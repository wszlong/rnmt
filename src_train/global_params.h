
using namespace std;
typedef float precision;
//typedef double precision;

struct attention_params {
	bool attention_model = true;
	bool feed_input = false;
	bool dump_alignments = false;
	string tmp_alginment_file = "NULL";
	string alignment_file = "alignment.txt";
};

struct global_params {
	
	//adam
	bool ADAM = false;
	precision alpha_adam = 0.001;
	precision beta_1 = 0.9;
	precision beta_2 = 0.999;
	precision epsilon = 1.0e-8;

	int sgd_epochs = 4;
	bool sgd_flag = true;	

	string unique_dir = "NULL";

	string load_model_name = "NULL";
	bool load_model_train = false;

	//for dropout
	bool dropout = false;
	precision dropout_rate = 1.0;

	bool random_seed = false;
	int random_seed_int = -1;
	
	string tmp_location = "";

	attention_params attent_params;
	
	bool clip_gradient = true;
	precision norm_clip = 5.0;

	bool softmax = true;

	//General settings
	static const bool debug = false;
	//static const bool debug = true;
	bool train = true;
	bool train_perplexity = true;
	bool shuffle = true;

	int minibatch_size = 16;
	int num_epochs = 10;
	precision learning_rate = 0.1;

	bool learning_rate_schedule = false;
	precision decrease_factor = 0.5;
	int half_way_count = -1; //What is the total number of words that mark half an epoch
	
	double margin = 0.0; 
	string dev_source_file_name;
	string dev_target_file_name;

	int source_vocab_size = -1;
	int target_vocab_size = -1;
	int Embedding_size = 100;
	int LSTM_size = 200;
	int num_layers = 1;
	vector<int> gpu_indicies;

	int screen_print_rate=5;

	//for saving the best model for training
	string best_model_file_name;
	bool best_model=false;
	double best_model_perp = DBL_MAX;
	
	string source_file_name;
	string target_file_name;
	int longest_sent = 100;
	
	string train_file_name = "NULL";
	int train_num_lines_in_file = -1;
	int train_total_words = -1;
	
	string test_file_name = "NULL";
	int test_num_lines_in_file = -1;
	int test_total_words = -1;

	string output_vocab_file = "vocab.nn";
	string input_weight_file = " model.nn";
	string output_weight_file = "model.nn";
	
	//?
	bool print_score = false; //Whether to print the score of the hypotheses or not

};
