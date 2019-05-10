
template<typename dType>
void neuralMT_model<dType>::initModel(int Embedding_size,int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
	int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
	string input_weight_file,string output_vocab_file,string output_weight_file,bool train_perplexity,int num_layers,
	vector<int> gpu_indicies,bool dropout,dType dropout_rate,attention_params attent_params,global_params &params) 
{
	
	//allocate GPU
	if(gpu_indicies.size()!=0) {
		if(gpu_indicies.size()!= num_layers+1) {
			cout<<"ERROR: multi gpu indicies you specified are invalid. There must be one index for each layer, plus one index for the softmax\n";
			exit (EXIT_FAILURE);
		}
	}	

	vector<int> final_gpu_indicies; // what layer is on what GPU
	if(gpu_indicies.size()!=0){
		final_gpu_indicies = gpu_indicies;
	}
	else {
		for(int i=0; i<num_layers+1; i++) {
			final_gpu_indicies.push_back(0);
		}
	}

	unordered_map<int,layer_gpu_info> layer_lookups; //get the layer lookups for each GPU
	for(int i=0; i<final_gpu_indicies.size()-1; i++) {
		if(layer_lookups.count(final_gpu_indicies[i])==0) {
			layer_gpu_info temp_layer_info;
			temp_layer_info.init(final_gpu_indicies[i]);
			layer_lookups[final_gpu_indicies[i]] = temp_layer_info;
		}
	}
	
	input_layer_source.ih_layer_info = layer_lookups[final_gpu_indicies[0]];
	input_layer_source_bi.ih_layer_info = layer_lookups[final_gpu_indicies[0]]; // bidirection
	input_layer_target.ih_layer_info = layer_lookups[final_gpu_indicies[0]];

	//Initialize the softmax layer
	softmax = new softmax_layer<dType>();
	s_layer_info = softmax->gpu_init(final_gpu_indicies.back());
	softmax->init_loss_layer(this,params);
	
	//Now print gpu info
	cout << "----------Memory status after loss (softmax/NCE) layer was initialized-----------\n";
	//print_GPU_Info();
	
	//bidirection 
	//input_layer_source_bi.init_Input_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,source_vocab_size,
	//	longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout,dropout_rate,true,params,true);
	
	//Initialize the input layer
	input_layer_source.init_Input_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,source_vocab_size,
		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout,dropout_rate,false,params,true);
	
	//bidirection 
	input_layer_source_bi.init_Input_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,source_vocab_size,
		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout,dropout_rate,true,params,true);
	
	input_layer_target.init_Input_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102,dropout,dropout_rate,false,params,false);

	this->input_weight_file = input_weight_file;
	this->output_vocab_file = output_vocab_file;
	this->output_weight_file = output_weight_file;
	this->debug = debug;
	train_perplexity_mode = train_perplexity;
	this->attent_params = attent_params;
	
	//CUDA_GET_LAST_ERROR("test0.1 initModel");

	cout << "----------------Memory status after Layer 1 was initialized----------------------\n";
	//print_GPU_Info();

	//do this to be sure addresses stay the same
	for(int i=1; i<num_layers; i++) {
		source_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
	}


	//now initialize hidden layers
	for(int i=1; i<num_layers; i++) {	

		source_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
		source_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,
			norm_clip,this,103,dropout,dropout_rate,i,params);
		

		target_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
		target_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(Embedding_size,LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,
			norm_clip,this,103,dropout,dropout_rate,i,params);

		cout << "-----------------Memory status after Layer " << i+1 << " was initialized-------------\n";
		//print_GPU_Info();
	}


	//initialize the attention layer on top layer, by this time all the other layers have been initialized
	if(attent_params.attention_model) {
		
		target_hidden_layers[num_layers-2].init_attention(final_gpu_indicies[num_layers-1],attent_params.feed_input,this,params);
		for(int i=0; i<longest_sent; i++) {
			
			target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_h_t = target_hidden_layers[num_layers-2].nodes[i].d_h_t;
			target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_d_ERRt_ht_tild  = target_hidden_layers[num_layers-2].nodes[i].d_d_ERRt_ht;
			target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_indicies_mask  = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01; //attention layer indicies
		}
		
		if(attent_params.feed_input) {
			input_layer_target.init_feed_input(&target_hidden_layers[num_layers-2]);
			input_layer_target.ih_layer_info.attention_forward = target_hidden_layers[num_layers-2].attent_layer->layer_info.forward_prop_done;
			target_hidden_layers[num_layers-2].attent_layer->layer_info.error_htild_below = input_layer_target.ih_layer_info.error_htild_below; //?
			}
	}

	cout << "-----------------Memory status after Attention Layer was initialized------------------\n";
	//print_GPU_Info();

	//!!define the relation among layers
	input_layer_source.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
	input_layer_source_bi.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]); //bidi
	
	input_layer_target.upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[0]); 
	
	for(int i=0; i<target_hidden_layers.size(); i++) {	
		//lower transfer stuff
		if(i==0) {
			//source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL); 
			source_hidden_layers[0].lower_layer.init_lower_transfer_layer_bi(true,true,&input_layer_source,&input_layer_source_bi,NULL); 
			target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_target,NULL);
		}
		else {
			source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
			target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i-1]);
		}
		

		//upper transfer stuff
		if(i==target_hidden_layers.size()-1) {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,false,softmax,NULL);
			softmax->init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i]); //softmax	
		}
		else {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[i+1]);
		}
	}
	
}


template<typename dType>
void neuralMT_model<dType>::initFileInfo(struct file_helper *file_info) {
	this->file_info = file_info;
}


template<typename dType>
void neuralMT_model<dType>::print_GPU_Info() {

	int num_devices = -1;
	cudaGetDeviceCount(&num_devices);
	size_t free_bytes, total_bytes = 0;
  	//int selected = 0;
  	for (int i = 0; i < num_devices; i++) {
	    cudaDeviceProp prop;
	    cudaGetDeviceProperties(&prop, i);
	    cout << "Device Number: " << i << "\n";
	    cout << "Device Name: " << prop.name << "\n";
	   	cudaSetDevice(i);
	    cudaMemGetInfo( &free_bytes, &total_bytes);
	    cout << "Total Memory (MB): " << (double)total_bytes/(1.0e6) << "\n";
	    cout << "Memory Free (MB): " << (double)free_bytes/(1.0e6) << "\n\n";
  	}
  	cudaSetDevice(0);
}


template<typename dType>
void neuralMT_model<dType>::compute_gradients(int *h_input_vocab_indicies_source,int *h_input_vocab_indicies_source_bi,
	int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,
	int *h_output_vocab_indicies_target,int current_source_length,int current_target_length,
	int *h_input_vocab_indicies_source_Wgrad,int *h_input_vocab_indicies_target_Wgrad,
	int len_source_Wgrad,int len_target_Wgrad,int *h_batch_info,file_helper *temp_fh,int time_index) 
{


	//std::cout << "----------------------STARTING COMPUTE GRADIENTS----------------------\n";
	train = true;
	source_length = current_source_length;

	//Send the CPU vocab input data to the GPU layers
	input_layer_source.prep_GPU_vocab_indices(h_input_vocab_indicies_source,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
	
	//bidirection
	//bi_dir_source.longest_sent_minibatch = current_source_length;//
	//bi_dir_source.reverse_indicies(h_input_vocab_indicies_source,current_source_length);
	//input_layer_source_bi.prep_GPU_vocab_indices(bi_dir_source.h_source_indicies,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
	//input_layer_source_bi.prep_GPU_vocab_indices_bi(h_input_vocab_indicies_source,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
	input_layer_source_bi.prep_GPU_vocab_indices(h_input_vocab_indicies_source_bi,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
	
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_source,current_source_length);
	}
	
	input_layer_target.prep_GPU_vocab_indices(h_input_vocab_indicies_target,h_input_vocab_indicies_target_Wgrad,current_target_length,len_target_Wgrad);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_target,current_target_length);
	}

	softmax->prep_GPU_vocab_indices(h_output_vocab_indicies_target,current_target_length);
	
	if(attent_params.attention_model) {
		target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(h_batch_info);
	}

	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GPU");

	////////////////////////////////Forward Compute////////////////////////////
	
	//cout << "***************source forward compute********************"<<endl;
	/////////////////Do the source side forward pass
	input_layer_source.nodes[0].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full,
		input_layer_source.d_input_vocab_indices_01_full,input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector,current_source_length);
	input_layer_source.nodes[0].forward_prop();
	
		
	//bidirection
	input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
		input_layer_source_bi.d_input_vocab_indices_01_full,input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector,current_source_length);
	input_layer_source_bi.nodes[0].forward_prop();

	//for bidirection
	for(int i=1; i<current_source_length; i++) {
		int step = i*input_layer_source.minibatch_size;
		input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
			input_layer_source_bi.d_input_vocab_indices_01_full+step,
			input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t,current_source_length);
		input_layer_source_bi.nodes[i].forward_prop();
	}
	
	devSynchAll(); //!!

	//mgpu stuff
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers[i].d_input_vocab_indices_01_full,
			source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector,current_source_length);
		source_hidden_layers[i].nodes[0].forward_prop();
	}

	for(int i=1; i<current_source_length; i++) {  //left to right
		int step = i*input_layer_source.minibatch_size;
		input_layer_source.nodes[i].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full+step,
			input_layer_source.d_input_vocab_indices_01_full+step,
			input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t,current_source_length);
		input_layer_source.nodes[i].forward_prop();
		

		//mgpu stuff
		for(int j=0; j<source_hidden_layers.size(); j++) { // down to up
			source_hidden_layers[j].nodes[i].update_vectors_forward_GPU(source_hidden_layers[j].d_input_vocab_indices_01_full+step,
				source_hidden_layers[j].nodes[i-1].d_h_t,source_hidden_layers[j].nodes[i-1].d_c_t,current_source_length);
			source_hidden_layers[j].nodes[i].forward_prop();
		}
	}
	

	//cout << "***************target forward compute********************"<<endl;
	////////////////Do the target side forward pass
	int prev_source_index = current_source_length-1;
	
	input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,input_layer_target.d_input_vocab_indices_01_full,
		input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t,current_target_length);

	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) { 
		target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
			source_hidden_layers[i].nodes[prev_source_index].d_h_t,source_hidden_layers[i].nodes[prev_source_index].d_c_t,current_target_length);
	}

	input_layer_target.nodes[0].forward_prop();
	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}
	
	//string test;
	//cin>>test;
	//cout<<"*********softmax 0******"<<endl;
	//mgpu stuff
	softmax->backprop_prep_GPU_mgpu(0);

	//cout<<"show softmax->d_output_vocab_indices_single: "<<endl;
	//show_matrix_int(softmax.d_output_vocab_indices_single, 128, 1);

	softmax->forward_prop(0);

	//cout<<"*********softmax 0 end******"<<endl;
	
	for(int i=1; i<current_target_length; i++) {  // left to right
		int step = i*input_layer_target.minibatch_size;
		input_layer_target.nodes[i].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t,current_target_length);
		input_layer_target.nodes[i].forward_prop();

		//mgpu stuff
		for(int j=0; j<target_hidden_layers.size(); j++) {  // down to up
			target_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
				target_hidden_layers[j].d_input_vocab_indices_01_full+step,
				target_hidden_layers[j].nodes[i-1].d_h_t,target_hidden_layers[j].nodes[i-1].d_c_t,current_target_length);
			target_hidden_layers[j].nodes[i].forward_prop();
		}

		//cout<<"*********softmax " <<i<<"******"<<endl;
		//mgpu stuff
		softmax->backprop_prep_GPU_mgpu(step);
		softmax->forward_prop(i);
	}

	devSynchAll();
	
	
	////////////////////////////////Backward Compute////////////////////////////////
	
	//cout << "***************target backward compute********************"<<endl;
	int last_index = current_target_length-1;

	int step = (current_target_length-1)*input_layer_target.minibatch_size;
	//softmax->backprop_prep_GPU(input_layer_target.nodes[last_index].d_h_t,step); //?

	//mgpu stuff
	softmax->backprop_prep_GPU_mgpu(step);
	softmax->back_prop1(current_target_length-1); // update d_h_t
	softmax->back_prop2(current_target_length-1); // update d_D and d_b

	//mgpu stuff ******* last column
	for(int i=target_hidden_layers.size()-1; i>=0; i--) {
		target_hidden_layers[i].nodes[last_index].backprop_prep_GPU(target_hidden_layers[i].d_init_d_ERRnTOtp1_ht,target_hidden_layers[i].d_init_d_ERRnTOtp1_ct);//,
		target_hidden_layers[i].nodes[last_index].back_prop_GPU(last_index);
	}

	input_layer_target.nodes[last_index].backprop_prep_GPU(input_layer_target.d_init_d_ERRnTOtp1_ht,input_layer_target.d_init_d_ERRnTOtp1_ct);//,
	input_layer_target.nodes[last_index].back_prop_GPU(last_index);
		
	
	for(int i=current_target_length-2; i>=0; i--) { //right to left
		step = i*input_layer_target.minibatch_size;

		//softmax->backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,step); //??

		//mgpu stuff
		softmax->backprop_prep_GPU_mgpu(step);
		softmax->back_prop1(i);
		softmax->back_prop2(i);

		for(int j=target_hidden_layers.size()-1; j>=0; j--) { //up to down
			target_hidden_layers[j].nodes[i].backprop_prep_GPU(target_hidden_layers[j].d_d_ERRnTOt_htM1,target_hidden_layers[j].d_d_ERRnTOt_ctM1);//,
			target_hidden_layers[j].nodes[i].back_prop_GPU(i);
		}

		input_layer_target.nodes[i].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,input_layer_target.d_d_ERRnTOt_ctM1);
		input_layer_target.nodes[i].back_prop_GPU(i);

	}
		
		
	//cout << "***************source backward compute********************"<<endl;
	
	//int prev_source_index = current_source_length-1;
	prev_source_index = current_source_length-1;

	//mgpu stuff ***** last column
	for(int i=source_hidden_layers.size()-1; i>=0; i--) {
	
		source_hidden_layers[i].nodes[prev_source_index].backprop_prep_GPU(target_hidden_layers[i].d_d_ERRnTOt_htM1,target_hidden_layers[i].d_d_ERRnTOt_ctM1);
		source_hidden_layers[i].nodes[prev_source_index].back_prop_GPU(prev_source_index);
	}

	input_layer_source.nodes[prev_source_index].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,input_layer_target.d_d_ERRnTOt_ctM1);//,input_layer_source.d_zeros);
	input_layer_source.nodes[prev_source_index].back_prop_GPU(prev_source_index);


	for(int i=current_source_length-2; i>=0; i--) { // right ro left

		for(int j=source_hidden_layers.size()-1; j>=0; j--) { // up to down
			source_hidden_layers[j].nodes[i].backprop_prep_GPU(source_hidden_layers[j].d_d_ERRnTOt_htM1,source_hidden_layers[j].d_d_ERRnTOt_ctM1);//,
			source_hidden_layers[j].nodes[i].back_prop_GPU(i);
		}
		input_layer_source.nodes[i].backprop_prep_GPU(input_layer_source.d_d_ERRnTOt_htM1,input_layer_source.d_d_ERRnTOt_ctM1);
		input_layer_source.nodes[i].back_prop_GPU(i);
	}

	devSynchAll(); //!!

	//**************** for bidirection encoder ******************/
	
	input_layer_source_bi.nodes[prev_source_index].backprop_prep_GPU(input_layer_source_bi.d_init_d_ERRnTOtp1_ht,input_layer_source_bi.d_init_d_ERRnTOtp1_ct);
	input_layer_source_bi.nodes[prev_source_index].back_prop_GPU(prev_source_index);
	for(int i=current_source_length-2; i>=0; i--) { 
		input_layer_source_bi.nodes[i].backprop_prep_GPU(input_layer_source_bi.d_d_ERRnTOt_htM1,input_layer_source_bi.d_d_ERRnTOt_ctM1);
		input_layer_source_bi.nodes[i].back_prop_GPU(i);
	}	
	
	if(debug) {
		cout<<"debug: "<<endl;
		grad_check_flag = true;
		dType epsilon =(dType)1e-4;
		devSynchAll();
		//src_fh_test = &src_fh;
		check_all_gradients(epsilon);
		grad_check_flag = false;
	}
	
	//cout<<"test"<<endl;
	
	//devSynchAll(); //add
	
	//Update the model parameter weights
	update_weights(time_index);

	clear_gradients();

	devSynchAll();

	if(train_perplexity_mode) {
		train_perplexity += softmax->get_train_perplexity();
	}


}


template<typename dType>
void neuralMT_model<dType>::check_all_gradients(dType epsilon) 
{
	devSynchAll();
	
	cout << "------------------CHECKING GRADIENTS ON SOURCE SIDE------------------------\n";
	input_layer_source.check_all_gradients(epsilon);
	
	cout << "------------------CHECKING GRADIENTS FOR BIDIRECTION SOURCE SIDE------------------------\n";
	input_layer_source_bi.check_all_gradients(epsilon);
	
	
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].check_all_gradients(epsilon);
	}
	
	cout << "------------------CHECKING GRADIENTS ON TARGET SIDE------------------------\n";
	input_layer_target.check_all_gradients(epsilon);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].check_all_gradients(epsilon);
	}
	softmax->check_all_gradients(epsilon);
	
}


template<typename dType>
void neuralMT_model<dType>::clear_gradients() {
	devSynchAll();
	
	input_layer_source.clear_gradients(false);
	input_layer_source_bi.clear_gradients(false); //bidirection
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].clear_gradients(false);
	}
	
	input_layer_target.clear_gradients(false);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].clear_gradients(false);
	}
	
	softmax->clear_gradients();

	devSynchAll();
}
	
	//Update the model parameters
template<typename dType>
void neuralMT_model<dType>::update_weights(int time_index) {

	devSynchAll();

	softmax->update_weights(time_index);
	
	//deniz::source_side = true; ??
	input_layer_source_bi.update_weights(time_index);//bidirection   // for same word embedding
	input_layer_source.update_weights(time_index);
	//input_layer_source_bi.update_weights();//bidirection

	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].update_weights(time_index);
	}

	//deniz::source_side = false;
	input_layer_target.update_weights(time_index);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].update_weights(time_index);
	}

	devSynchAll();
	if(attent_params.attention_model) {
		source_hidden_layers[source_hidden_layers.size()-1].zero_attent_error();	
	}
	
	devSynchAll();
}


template<typename dType>
double neuralMT_model<dType>::getError(){
	
	//cout << "------------------------GET ERROR STARTING-----------------------------\n";
	double loss=0;

	source_length = file_info->current_source_length;

	
	//source indicies prepare
	input_layer_source.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_source_Wgrad,
		file_info->current_source_length,file_info->len_source_Wgrad);
	
	//bidirection
	//input_layer_source_bi.prep_GPU_vocab_indices_bi(file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_source_Wgrad,
	//file_info->current_source_length,file_info->len_source_Wgrad);
	
	input_layer_source_bi.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source_bi,file_info->h_input_vocab_indicies_source_Wgrad,
	file_info->current_source_length,file_info->len_source_Wgrad);
	
		
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source,file_info->current_source_length);
	}

	//target indicies prepare
	input_layer_target.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_target,file_info->h_input_vocab_indicies_target_Wgrad,
		file_info->current_target_length,file_info->len_target_Wgrad);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_target,file_info->current_target_length);
	}
	
	//softmax indicies prepare
	softmax->prep_GPU_vocab_indices(file_info->h_output_vocab_indicies_target,file_info->current_target_length);
	
	if(attent_params.attention_model) {
		target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(file_info->h_batch_info);
	}

	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GETERROR");


	//cout << "**********************source forward compute***************************\n";
	
	input_layer_source.nodes[0].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full,
		input_layer_source.d_input_vocab_indices_01_full,
		input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector,file_info->current_source_length);
	input_layer_source.nodes[0].forward_prop();
	
	
	//bidirection
	input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
		input_layer_source_bi.d_input_vocab_indices_01_full,
		input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector,file_info->current_source_length);
	input_layer_source_bi.nodes[0].forward_prop();

	for(int i=1; i<file_info->current_source_length; i++) { 
		int step = i*input_layer_source.minibatch_size;
		input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
			input_layer_source_bi.d_input_vocab_indices_01_full+step,
			input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t,file_info->current_source_length);
		input_layer_source_bi.nodes[i].forward_prop();
	}
	
	devSynchAll(); //!!

	//mgpu stuff
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers[i].d_input_vocab_indices_01_full,
			source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector,file_info->current_source_length);
		source_hidden_layers[i].nodes[0].forward_prop();
	}

	
	for(int i=1; i<file_info->current_source_length; i++) { //left to right
		int step = i*input_layer_source.minibatch_size;
		input_layer_source.nodes[i].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full+step,
			input_layer_source.d_input_vocab_indices_01_full+step,
			input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t,file_info->current_source_length);
		input_layer_source.nodes[i].forward_prop();

		//mgpu stuff
		for(int j=0; j<source_hidden_layers.size(); j++) { // down to up
			source_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
				source_hidden_layers[j].d_input_vocab_indices_01_full+step,
				source_hidden_layers[j].nodes[i-1].d_h_t,source_hidden_layers[j].nodes[i-1].d_c_t,file_info->current_source_length);
			source_hidden_layers[j].nodes[i].forward_prop();
		}
	}


	//cout << "**********************target forward compute***************************\n";
	int prev_source_index = file_info->current_source_length-1;
	input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
		input_layer_target.d_input_vocab_indices_01_full,
		input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t,file_info->current_target_length);

	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(
			target_hidden_layers[i].d_input_vocab_indices_01_full,
			source_hidden_layers[i].nodes[prev_source_index].d_h_t,source_hidden_layers[i].nodes[prev_source_index].d_c_t,file_info->current_target_length);
	}
	
	input_layer_target.nodes[0].forward_prop();
	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}
	devSynchAll();

	//cout<<"test 1 getError"<<endl;

	//softmax->backprop_prep_GPU(input_layer_target.nodes[0].d_h_t,0);
	softmax->backprop_prep_GPU_mgpu(0);
	
	//cout<<"test 2 getError"<<endl;
	
	loss += softmax->compute_loss_GPU(0);	
	
	//cout<<"test 3 getError"<<endl;
	
	devSynchAll();
	
	for(int i=1; i<file_info->current_target_length; i++) { //left to right
		int step = i*input_layer_target.minibatch_size;

		input_layer_target.nodes[i].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,
			input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t,file_info->current_target_length);
		
		input_layer_target.nodes[i].forward_prop();

		//mgpu stuff
		for(int j=0; j<target_hidden_layers.size(); j++) { //down to up
			target_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
				target_hidden_layers[j].d_input_vocab_indices_01_full+step,
				target_hidden_layers[j].nodes[i-1].d_h_t,target_hidden_layers[j].nodes[i-1].d_c_t,file_info->current_target_length);
			target_hidden_layers[j].nodes[i].forward_prop();
		}

		devSynchAll();
		
		//softmax->backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,step);
		softmax->backprop_prep_GPU_mgpu(step);
		//cout<<"test 4 getError: "<<i<<endl;

		loss += softmax->compute_loss_GPU(i);
		
		//cout<<"test 5 getError: "<<i<<endl;
		
		devSynchAll();
	}
	
	return loss;
}

template<typename dType>
double neuralMT_model<dType>::get_perplexity(string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
	int source_vocab_size,int target_vocab_size,int &test_total_words) {

	file_helper file_info(test_file_name,minibatch_size,test_num_lines_in_file,longest_sent,
		source_vocab_size,target_vocab_size,test_total_words); //Initialize the file information
	initFileInfo(&file_info);

	//file_helper_source perp_fhs;

	
	int current_epoch = 1;
	cout << "Getting perplexity of dev set\n";

	double P_data_GPU = 0;
	while(current_epoch <= 1) {
		
		bool success = file_info.read_minibatch();
		
		double temp = getError();
		P_data_GPU += temp;
	
		if(!success) {
			current_epoch+=1;
		}
	}
	
	P_data_GPU = P_data_GPU/log(2.0); 
	double perplexity_GPU = pow(2,-1*P_data_GPU/file_info.total_target_words);
	
	cout << "Total target words: " << file_info.total_target_words << "\n";
	cout <<  setprecision(15) << "Perplexity dev set: " << perplexity_GPU << "\n";
	cout <<  setprecision(15) << "P_data dev set: " << P_data_GPU << "\n";
	
	return perplexity_GPU;
}


template<typename dType>
void neuralMT_model<dType>::update_learning_rate(dType new_learning_rate) {

	input_layer_source.learning_rate = new_learning_rate;
	input_layer_source_bi.learning_rate = new_learning_rate; //bidirection
	input_layer_target.learning_rate = new_learning_rate;
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].learning_rate = new_learning_rate;
	}
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].learning_rate = new_learning_rate;
	}

	softmax->update_learning_rate(new_learning_rate);
}


template<typename dType>
void neuralMT_model<dType>::adam_switch_sgd() {

	input_layer_source.ADAM = false;
	input_layer_source_bi.ADAM = false; //bidirection
	input_layer_target.ADAM = false;
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].ADAM = false;
	}
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].ADAM = false;
	}
	
	target_hidden_layers[target_hidden_layers.size()-1].attent_layer->ADAM = false;

	softmax->adam_switch_sgd();
}


template<typename dType>
void neuralMT_model<dType>::dump_best_model(std::string best_model_name,std::string const_model) {

    cout << "Writing model file " << best_model_name  << "\n";

	if(boost::filesystem::exists(best_model_name)) {
		boost::filesystem::remove(best_model_name);
	}

	//ifstream const_model_stream;
	//const_model_stream.open(const_model.c_str());

	ofstream best_model_stream;
	best_model_stream.open(best_model_name,ios_base::out | ios_base::binary);

	//best_model_stream.precision(numeric_limits<dType>::digits10 + 2); //!?

	/***
	//now create the new model file
	//string str;
	//string word;
	//getline(const_model_stream, str);
	//best_model_stream << str << "\n";
	//getline(const_model_stream, str);
	//best_model_stream << str << "\n";
	
	while(getline(const_model_stream, str)) {
		//best_model_stream << str << "\n";
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
		}
	}

	while(getline(const_model_stream, str)) {
		//best_model_stream << str << "\n";
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
		}
	}

	***/

	input_layer_source.dump_weights(best_model_stream);
	input_layer_source_bi.dump_weights(best_model_stream); //bidirection

	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].dump_weights(best_model_stream);
	}
	
	input_layer_target.dump_weights(best_model_stream);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].dump_weights(best_model_stream);
	}
	
	softmax->dump_weights(best_model_stream);
	
	best_model_stream.flush();
	best_model_stream.close();
	//const_model_stream.close();

}

template<typename dType>
void neuralMT_model<dType>::dump_weights() {

    //if(MY_CUDA::cont_train) {
       	/*** 
		ifstream tmp_input(output_weight_file.c_str());
        tmp_input.clear();
        tmp_input.seekg(0, ios::beg);
    	string str;
        vector<std::string> beg_lines; 
    	getline(tmp_input, str);
        beg_lines.push_back(str);
    	getline(tmp_input, str);
        beg_lines.push_back(str);
    	
		while(getline(tmp_input, str)) {
            beg_lines.push_back(str);
    		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
    				break; //done with source mapping
    		}
    	}
		
		while(std::getline(tmp_input, str)) {
			beg_lines.push_back(str); 
			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with target mapping
			}
		}
        
        long pos = tmp_input.tellg();
       
        tmp_input.close();
        output.open(output_weight_file.c_str());
	    output.precision(std::numeric_limits<dType>::digits10 + 2);
        output.clear();
        output.seekp(0,std::ios_base::beg); //output.seekp(pos,std::ios_base::beg);
        
		for(int i=0; i<beg_lines.size(); i++) {
            output << beg_lines[i] << "\n";    
        }	
		***/
	
		//if(boost::filesystem::exists(output_weight_file)) {
		//	boost::filesystem::remove(output_weight_file);
		//}

        //output.open(output_weight_file,ios_base::out | ios_base::binary);
        //output.clear();
        //output.seekp(0,std::ios_base::beg); //output.seekp(pos,std::ios_base::beg);

	//}
	//else {

        //output.open(output_weight_file,ios_base::out | ios_base::binary);
		//output.open(output_weight_file.c_str(),std::ios_base::app);
	    //output.precision(std::numeric_limits<dType>::digits10 + 2);
	//}

	if(MY_CUDA::dump_after_nEpoch) {
		
		string save_model_name;
		save_model_name += "save_models_"+std::to_string(MY_CUDA::curr_dump_num)+".nn";
		MY_CUDA::curr_dump_num += 0.5;
	
		cout << "Writing model file  " << save_model_name  << "\n";
		
		if(boost::filesystem::exists(save_model_name)) {
			boost::filesystem::remove(save_model_name);
		}
		
		output.open(save_model_name,ios_base::out | ios_base::binary);
	}
	else {
		
		if(boost::filesystem::exists(output_weight_file)) {
			boost::filesystem::remove(output_weight_file);
		}

        output.open(output_weight_file,ios_base::out | ios_base::binary);
	
	}


	
	input_layer_source.dump_weights(output);
	input_layer_source_bi.dump_weights(output); //bidirection
	
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].dump_weights(output);
	}
	
	input_layer_target.dump_weights(output);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].dump_weights(output);
	}
	
	softmax->dump_weights(output);

	output.close();
}


//Load in the weights from a file, so the model can be used
template<typename dType>
void neuralMT_model<dType>::load_weights() {
	
	/***	
	ifstream input_vocab;	
	input_vocab.open(output_vocab_file.c_str());

	//now load the weights by bypassing the intro stuff
	string str;
	string word;
	getline(input_vocab, str);
	getline(input_vocab, str);
	while(getline(input_vocab, str)) {
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
		}
	}

	while(std::getline(input_vocab, str)) {
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
		}
	}

	input_vocab.close();
	***/

	ifstream input_model;  //model weight
	input_model.open(input_weight_file, ios_base::in | ios_base::binary);

	input_layer_source.load_weights(input_model);
	input_layer_source_bi.load_weights(input_model);//bi

	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].load_weights(input_model);
	}
	
	input_layer_target.load_weights(input_model);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].load_weights(input_model);
	}

	softmax->load_weights(input_model);

	input_model.close();
}

