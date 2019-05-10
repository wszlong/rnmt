
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cudnn.h>

//Boost
#include "boost/program_options.hpp" 
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

//My own includes
#include "add_model_info.h"
#include "global_params.h"
#include "input_file_prep.h"
#include "CUDA_UTIL.h"
#include "gpu_info_struct.h"
#include "custom_kernels.h"
#include "model.h"
#include "fileHelper.h"
#include "Util.h"
#include "model.hpp"

#include "LSTM.h" //add
#include "LSTM_HH.h"
#include "softmax.h" //add
#include "Input_To_Hidden_Layer.h" // add
#include "Hidden_To_Hidden_Layer.h"
#include "attention_layer.h"
#include "attention_node.h"

#include "LSTM.hpp"
#include "LSTM_HH.hpp"
#include "softmax.hpp"
#include "Input_To_Hidden_Layer.hpp"
#include "Hidden_To_Hidden_Layer.hpp"
#include "attention_layer.hpp"
#include "attention_node.hpp"

using namespace std;

void command_line_parse(global_params &params, int argc, char **argv) {

	vector<string> train_files;
	vector<string> adaptive_learning_rate;
	vector<string> cont_train;
	
	vector<precision> adam_files;

	vector<int> gpu_indicies;
	//vector<precision> clip_cell_vals;

	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Run to get help on how to use the program.")
		("train,t", po::value<vector<string>> (&train_files)->multitoken(), "Train a model with inputdata files and a name for the neural network output file. \nFormat:<source file name> <target file name> <neural network output name>")
		("cont-train,C", po::value<vector<string>> (&cont_train)->multitoken(), "Resume training of a model")
		("num-layers,N", po::value<int> (&params.num_layers), "set the number of LSTM layers. Default: 2")
		("multi-gpu,M", po::value<vector<int>> (&gpu_indicies)->multitoken(), "Train the model on multiple gpus. \nFormat: <gpu for layer 1> <gpu for layer 2> ... <gpu for softmax>\n, Default: all layers and softmax lie on gpu 0")
		("tmp-dir-location", po::value<string> (&params.tmp_location), "a tmp directiory must be created for data preparation. Default: Current directory")
		("learning-rate,l", po::value<precision> (&params.learning_rate), "Set the learning rate. Default: 0.1")
		("dropout,d", po::value<precision> (&params.dropout_rate), "Use dropout and set the dropout rate. Default: 1.0")
		("random-seed", po::value<int> (&params.random_seed_int), "Specify a random seed, instead of the model being seeded with the current time")
		("longest-sent,L", po::value<int> (&params.longest_sent), "Set the maximum sentence length. Default: 100")
		("embedding-size,E", po::value<int> (&params.Embedding_size), "Set the embedding size. Default: 100")
		("hiddenstate-size,H", po::value<int> (&params.LSTM_size), "Set hiddenstate size. Default: 100")
		("feed-input", po::value<bool> (&params.attent_params.feed_input), "Bool for weather feed input. Default: False")
		
  		("source-vocab-size,v",po::value<int>(&params.source_vocab_size),"Set source vocab size. Default: number of unique words in source training corpus")
  		("target-vocab-size,V",po::value<int>(&params.target_vocab_size),"Set target vocab size. Default: number of unique words in target training corpus")
		("shuffle", po::value<bool> (&params.shuffle),"true if you want to shuffle the train data. Default: True")
		("number-epochs,n", po::value<int> (&params.num_epochs), "Set number os epoch. Default: 10")
		("adaptive-halve-lr,a", po::value<vector<string>> (&adaptive_learning_rate)->multitoken(), "change the rate, "\
		 	" when the perplexity on your specified dev set increases from the previous half epoch by some constant, so "\
			" new_learning_rate = constant*old_learning rate, by default the constant is 0.5."\
			"\nFormat:  <source dev file name> <target dev file name>")
		("adaptive-decrease-factor,A", po::value<precision> (&params.decrease_factor),"To be used with adaptive-halve-lr. Default: 0.5")

		//("clip-cell", po::value<vector<precision>> (&clip_cell_vals)->multitoken(), "Specify the cell clip threshold")
		("minibatch_size,m", po::value<int> (&params.minibatch_size), "Set minibatch size. Defalut: 16")
		("best-model,B", po::value<string> (&params.best_model_file_name)->multitoken(), "During train save the best model")
		("save-after-n-epoch",po::value<float>(&MY_CUDA::begin_dump_num),"Save the every model after n epochs")
		("print-score", po::value<bool> (&params.print_score), "print the log prob")
		("adam", po::value<vector<precision>> (&adam_files)->multitoken(), "use adam algorithm instead of SGD");
	po::variables_map vm;
	try{
		
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if(vm.count("help")) {
			cout<<"********************************************************************************"<<endl;
			cout<<"********************Cuda Based C++ Code for NMT Train Process*******************"<<endl;
			cout<<desc<<endl;
			exit(EXIT_FAILURE);
		}

		if(vm.count("random-seed")) {
			params.random_seed = true;
		}

		if (vm.count("tmp-dir-location")) {
		  if (params.tmp_location != "") {
			if (params.tmp_location[params.tmp_location.size()-1]!='/') {
			  params.tmp_location+="/";
			}
		  }
		}

		if(vm.count("shuffle")) {
			MY_CUDA::shuffle_data = params.shuffle;
		}

		if(vm.count("cont-train")) {
			MY_CUDA::cont_train = true;
		}
		else {
			MY_CUDA::cont_train = false;
		}

		if(vm.count("save-after-n-epoch")) {
			MY_CUDA::dump_after_nEpoch = true;
			MY_CUDA::curr_dump_num = MY_CUDA::begin_dump_num;
		}
		
		if(vm.count("adam")) {
			params.ADAM = true;
			params.alpha_adam = adam_files[0];
			params.sgd_epochs = adam_files[1];
			params.sgd_flag = false;
			//cout<<"alpha_adam: "<<params.alpha_adam<<endl;
			//cout<<"sgd_epochs: "<<params.sgd_epochs<<endl;
		}
		

		if(vm.count("train") || vm.count("cont-train")) {
			

			boost::filesystem::path unique_path = boost::filesystem::unique_path();
			if(vm.count("tmp-dir-location")) {
				unique_path = boost::filesystem::path(params.tmp_location + unique_path.string());
			}
			
			cout<<"Temp diretory being created named: "<<unique_path.string()<<"\n";
			boost::filesystem::create_directories(unique_path);
			params.unique_dir = unique_path.string();
			params.train_file_name = params.unique_dir + "/train.txt";
		
			if(vm.count("dropout")) {
				params.dropout = true;
			}	

			if(vm.count("multi-gpu")) {
				params.gpu_indicies = gpu_indicies;
				MY_CUDA::gpu_indicies = gpu_indicies;
			}


			if(vm.count("cont-train")) {

				
				params.source_file_name = cont_train[0];
				params.target_file_name = cont_train[1];
				params.output_vocab_file = cont_train[2];
				params.input_weight_file = cont_train[3];
				params.output_weight_file = cont_train[3];
				params.load_model_train = true;
				params.load_model_name = params.input_weight_file;
				cout<<"Load model name: "<<params.load_model_name<<"\n";
				
				input_file_prep input_helper;

				input_helper.integerize_file_nmt(params.output_vocab_file, params.source_file_name, params.target_file_name, params.train_file_name,
					params.longest_sent, params.minibatch_size, params.Embedding_size, params.LSTM_size, params.source_vocab_size, params.target_vocab_size, params.num_layers);

			}
			// not cont-train
			else { 

				params.source_file_name = train_files[0];
				params.target_file_name = train_files[1];
				params.output_vocab_file = train_files[2]; //vocab file
				params.output_weight_file = train_files[3]; //final weight file

				input_file_prep input_helper;
				
				bool success = true;
				success = input_helper.prep_files_train_nmt(params.minibatch_size, params.longest_sent, params.source_file_name, params.target_file_name, 
					params.train_file_name, params.source_vocab_size, params.target_vocab_size,
					params.shuffle, params.output_vocab_file, params.Embedding_size, params.LSTM_size, params.num_layers);
				
				if(!success) {
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit(EXIT_FAILURE);
				
				}			
			}

			if(vm.count("adaptive-halve-lr")) {
				params.learning_rate_schedule = true;

				params.dev_source_file_name = adaptive_learning_rate[0];
				params.dev_target_file_name = adaptive_learning_rate[1];
				params.test_file_name = params.unique_dir + "/validation.txt";

				input_file_prep input_helper;
				input_helper.integerize_file_nmt(params.output_vocab_file, params.dev_source_file_name, params.dev_target_file_name, params.test_file_name,
					params.longest_sent, params.minibatch_size, params.Embedding_size, params.LSTM_size, params.source_vocab_size, params.target_vocab_size, params.num_layers);

			}

			if(vm.count("best-model")) {
				params.best_model = true;
			}

			add_model_info(params.num_layers, params.Embedding_size, params.LSTM_size, params.target_vocab_size, params.source_vocab_size, params.attent_params.feed_input, params.output_vocab_file);
			params.train = true;
			return;

		}
		else {
			cout<<"Error: you must eigth train or continue training for NMT, '-h' for help."<<endl;
			exit(EXIT_FAILURE);
		}

	}
	catch(po::error& e) {
		cout<<"ERROR: "<<e.what()<<endl;
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	
	//Timing stuff
	chrono::time_point<chrono::system_clock> start_total,start_total_decode,
	end_total, begin_minibatch,end_minibatch,begin_decoding,end_decoding,begin_epoch;
	
	chrono::duration<double> elapsed_seconds;
  	start_total = chrono::system_clock::now();
	
  	global_params params; 
	
	MY_CUDA::curr_seed = static_cast<unsigned int>(time(0));
	MY_CUDA::curr_seed = min((unsigned int)100000000,MY_CUDA::curr_seed);//to prevent overflow

	command_line_parse(params, argc, argv);
	
	//randomize the seed
	if(params.random_seed) {
    MY_CUDA::gen.seed(static_cast<unsigned int>(params.random_seed_int));
	}
  else {
    MY_CUDA::gen.seed(static_cast<unsigned int>(time(0)));
  }
	
  	neuralMT_model<precision> model;
	printIntroMessage(params);
	
	//!!init model
	model.initModel(params.Embedding_size,params.LSTM_size,params.minibatch_size,params.source_vocab_size,params.target_vocab_size,
		params.longest_sent,params.debug,params.learning_rate,params.clip_gradient,params.norm_clip,
		params.input_weight_file,params.output_vocab_file,params.output_weight_file,params.train_perplexity,
		params.num_layers,params.gpu_indicies,params.dropout,params.dropout_rate,params.attent_params,params);

	// load model for continue training
	if(params.load_model_train) {
		
		string temp_swap_weights = model.input_weight_file; //
		model.input_weight_file = params.load_model_name;
		model.load_weights();
		model.input_weight_file = temp_swap_weights;
	}


	//***********************Train the model******************************//
	if(params.train) {
		int curr_batch_num_SPEED = 0;
		const int thres_batch_num_SPEED = params.screen_print_rate;//set this to whatever
		int total_words_batch_SPEED = 0;
		double total_batch_time_SPEED = 0;

		//File info for the training file
		file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size,params.train_total_words); //Initialize the file information

		params.half_way_count = params.train_total_words/2;
		int current_epoch = 1;
		cout << "Starting model training\n";
		cout << "-----------------------------------"  << "\n";
		cout << "Starting epoch 1\n";
		cout << "-----------------------------------"  << "\n";

		int total_words = 0;
		precision temp_learning_rate = params.learning_rate;
		double old_perplexity = 0;
		bool learning_rate_flag = true;
		model.train_perplexity = 0;
		begin_epoch = chrono::system_clock::now();
		int time_index = 0;

		while(current_epoch <= params.num_epochs) {
			begin_minibatch = chrono::system_clock::now();
			bool success = file_info.read_minibatch();

			end_minibatch = chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;
			total_batch_time_SPEED += elapsed_seconds.count();
			begin_minibatch = chrono::system_clock::now();
			time_index += 1;

			model.initFileInfo(&file_info); //copy file_info to model->file_info
			
			//!!compute gradients and update params
			model.compute_gradients(file_info.h_input_vocab_indicies_source,file_info.h_input_vocab_indicies_source_bi,file_info.h_output_vocab_indicies_source,
				file_info.h_input_vocab_indicies_target,file_info.h_output_vocab_indicies_target,
				file_info.current_source_length,file_info.current_target_length,
				file_info.h_input_vocab_indicies_source_Wgrad,file_info.h_input_vocab_indicies_target_Wgrad,
				file_info.len_source_Wgrad,file_info.len_target_Wgrad,file_info.h_batch_info,&file_info,time_index);
			
			end_minibatch = std::chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;

			total_batch_time_SPEED+= elapsed_seconds.count();
			total_words_batch_SPEED+=file_info.words_in_minibatch;
			
			if(curr_batch_num_SPEED>=thres_batch_num_SPEED) {
				cout << "Time to compute gradients for previous " << params.screen_print_rate << " minibatches: "  << total_batch_time_SPEED/60.0 << " minutes\n";
				cout << "Number of words in previous " << params.screen_print_rate << " minibatches: "  << total_words_batch_SPEED << "\n";
				cout << "Throughput for previous " << params.screen_print_rate << " minibatches: " << (total_words_batch_SPEED)/(total_batch_time_SPEED) << " words per second\n";
				cout << total_words << " words out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
				
				total_words_batch_SPEED = 0;
				total_batch_time_SPEED = 0;
				curr_batch_num_SPEED = 0;

			}
			curr_batch_num_SPEED++;
			total_words += file_info.words_in_minibatch;

			//stuff for perplexity based learning schedule
			if(params.learning_rate_schedule && total_words>=params.half_way_count && learning_rate_flag) {
			
				//save model after n epochs
				if(MY_CUDA::dump_after_nEpoch && current_epoch > MY_CUDA::begin_dump_num) {
					devSynchAll();
					model.dump_weights();
				}

				learning_rate_flag = false;
				double new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
					params.source_vocab_size,params.target_vocab_size,params.test_total_words);
				cout << "Old dev set Perplexity: " << old_perplexity << "\n";
				cout << "New dev set Perplexity: " << new_perplexity << "\n";
				
				if ( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1 && params.sgd_flag) {
					temp_learning_rate = temp_learning_rate*params.decrease_factor;
					model.update_learning_rate(temp_learning_rate);
					
					//cout<<"New alpha_adam: "<<model.input_layer_source.alpha_adam <<endl;
					cout << "New learning rate:" << temp_learning_rate <<"\n\n";
				}
				
				//perplexity is better so output the best model file
				if(params.best_model && params.best_model_perp > new_perplexity) {
					model.dump_best_model(params.best_model_file_name,params.output_weight_file);
					params.best_model_perp = new_perplexity;
				}	
				old_perplexity = new_perplexity;
			}

			if(!success) {  // next epoch or over

				//save model after n epochs
				if(MY_CUDA::dump_after_nEpoch && current_epoch >= MY_CUDA::begin_dump_num) {
					devSynchAll();
					model.dump_weights();
				}


				current_epoch += 1;
				
				double new_perplexity;
				if(params.learning_rate_schedule) {
					new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
						params.source_vocab_size,params.target_vocab_size,params.test_total_words);
				}

				//stuff for perplexity based learning schedule
				if(params.learning_rate_schedule) {
					
					cout << "Old dev set Perplexity: " << old_perplexity << "\n";
					cout << "New dev set Perplexity: " << new_perplexity << "\n";
					
					if( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1 && params.sgd_flag) {
						temp_learning_rate = temp_learning_rate*params.decrease_factor;
						model.update_learning_rate(temp_learning_rate);
						cout << "New learning rate:" << temp_learning_rate <<"\n\n";
					}

					if(params.best_model && params.best_model_perp > new_perplexity) {
						model.dump_best_model(params.best_model_file_name,params.output_weight_file);
						params.best_model_perp = new_perplexity;
					}

					learning_rate_flag = true;
					old_perplexity = new_perplexity;
				}

				if(params.train_perplexity) {
					model.train_perplexity = model.train_perplexity/log(2.0);
					cout << "PData on train set: "  << model.train_perplexity << "\n";
					cout << "Total target words: " << file_info.total_target_words << "\n";
					cout << "Training set perplexity: " << pow(2,-1*model.train_perplexity/file_info.total_target_words) << "\n";
					model.train_perplexity = 0;
				}
				
				if(current_epoch >= params.sgd_epochs && params.ADAM) {  //params.sgd_epochs = 4

					model.adam_switch_sgd();
					if(!params.sgd_flag) {
						cout<<"***********begin using SGD************"<<"\n\n";
					}
					params.sgd_flag = true;
					params.ADAM = false;
				}

				total_words=0;
				if(current_epoch <= params.num_epochs) {
                    elapsed_seconds = chrono::system_clock::now() - begin_epoch;
          			cout << "Previous Epoch time (minutes): " << (double)elapsed_seconds.count()/60.0 << "\n";
          			begin_epoch = chrono::system_clock::now();
					cout << "-----------------------------------"  << "\n";
					cout << "Starting epoch " << current_epoch << "\n";
					cout << "-----------------------------------"  << "\n";
				}

			}
			devSynchAll();
		}
		//Now that training is done, dump the weights

		if (!MY_CUDA::dump_after_nEpoch) {
			devSynchAll();
			model.dump_weights();
		}
	}

	//Compute the final runtime
	end_total = chrono::system_clock::now();	
  	elapsed_seconds = end_total-start_total;
 	cout << "\n\n\n";
  	cout << "Total Program Runtime: " << (double)elapsed_seconds.count()/(60.0*60.0) << " hours" << "\n";
	return 0;
}

