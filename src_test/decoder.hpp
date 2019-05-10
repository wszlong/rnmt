

Decoder::Decoder(string model_file, string vocab_file, int beam_size, int gpu_num)
{
	
	this->gpu_num = gpu_num;
    this->beam_size = beam_size; //beamsearch sizei
	this->input_weight_file = model_file;
	this->input_vocab_file = vocab_file;
    
	this->B = 1;
    Tmax = 200;
    T = 200;
	
	//model params and dictionary !!!
	cout<<"Load Model and Init..."<<endl;;
	load_and_init_model(src_w2i,tgt_i2w);


}


void Decoder::load_and_init_model(map<string, int> &src_w2i, map<int, string> &tgt_i2w)
{
	
	//string input_weight_file = "best.new.2.nn";
	
	ifstream input_vocab;
	input_vocab.open(input_vocab_file.c_str());
	
	string str;
	string word;
	vector<string> file_model_info;

	getline(input_vocab, str); //model info   //2 1000 1000 30000 30000 1 
	istringstream iss(str, istringstream::in);
	while(iss >> word){
		file_model_info.push_back(word);
	}
	
	num_layers = stoi(file_model_info[0]); 
	Embedding_size = stoi(file_model_info[1]); //Embedding_size	
	LSTM_size = stoi(file_model_info[2]); // LSTM_size 1000
	target_vocab_size = stoi(file_model_info[3]); // 30000
	source_vocab_size = stoi(file_model_info[4]); // 30000
	
	if(stoi(file_model_info[5]) == 1) {
		feed_input = true;
	}

	getline(input_vocab, str); // ======
	
	//model layers structure
	for(int i=0; i<num_layers-1; i++){
		source_hidden_layers.push_back(Hidden_To_Hidden_Layer());
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer());
	}
			
	//model stucture
	init_model_structure();

	//!!! alloc params memory
	init_model();
		
	//source dict and target dict
	while(getline(input_vocab, str)){
		int tmp_index;
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='='){
			break;	
		} 

		istringstream iss(str, istringstream::in);
		iss >> word;
		tmp_index = stoi(word);
		iss >> word;
		src_w2i[word] = tmp_index;
	}
	//cout<<"src_w2i test: "<<src_w2i["中国"]<<endl;
	while(getline(input_vocab, str)){
		int tmp_index;	
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='='){
			break;
		}

		istringstream iss(str, istringstream::in);
		iss >> word;
		tmp_index = stoi(word);
		iss >> word;
		tgt_i2w[tmp_index] = word;
	}
	
	input_vocab.close();
	
	//cout <<"vocab file load end, start load weight file.."<<endl;	
	ifstream input_model;
	input_model.open(input_weight_file, ios_base::in | ios_base::binary);

	//source params	
	input_layer_source.load_weight(input_model);
	input_layer_source_bi.load_weight(input_model);
	
	cudaSetDevice(gpu_num);
	cudaMemcpy(input_layer_source_bi.d_W, input_layer_source.d_W, LSTM_size*source_vocab_size*sizeof(float), cudaMemcpyDeviceToDevice);

	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	
	for(int i=0; i<num_layers-1; i++){
		//cout<< "loda source_hidden_layers: "<< i<<endl;
		source_hidden_layers[i].load_weight(input_model);	
	}
		
	//target params
	input_layer_target.load_weight(input_model);
	for(int i=0; i<num_layers-1; i++){
		//cout<< "loda target_hidden_layers: "<< i<<endl;
		target_hidden_layers[i].load_weight(input_model);	
	}

	// attention and softmax params
	attention_softmax_target.load_weight(input_model);
   	
	input_model.close();
	
	//model stucture
	//init_model_structure();

}


void Decoder::init_model()
{
	//model param init
	input_layer_source.init_Input_To_Hidden_Layer(Embedding_size, LSTM_size, 1, source_vocab_size, false, gpu_num, false);
	
	input_layer_source_bi.init_Input_To_Hidden_Layer(Embedding_size, LSTM_size, 1, source_vocab_size, false, gpu_num, true);
	
	for(int i=0; i<num_layers-1; i++){
		source_hidden_layers[i].init_Hidden_To_Hidden_Layer(LSTM_size, 1, gpu_num);
	}
	
	
	input_layer_target.init_Input_To_Hidden_Layer(Embedding_size, LSTM_size, beam_size, target_vocab_size, feed_input, gpu_num, false);
	
	for(int i=0; i<num_layers-1; i++){
		target_hidden_layers[i].init_Hidden_To_Hidden_Layer(LSTM_size, beam_size, gpu_num);
	}

	attention_softmax_target.init_Attention_Softmax_Layer(Embedding_size, LSTM_size, beam_size, target_vocab_size, gpu_num);
	
}


void Decoder::init_model_structure() {

	input_layer_source.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
	input_layer_source_bi.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
	
	input_layer_target.upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[0]);

	for(int i=0; i<target_hidden_layers.size(); i++) {	
		//lower transfer stuff
		if(i==0) {
			source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL); 
			target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_target,NULL);
		}
		else {
			source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
			target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i-1]);
		}
	
		//upper transfer stuff
		if(i==target_hidden_layers.size()-1) {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,&attention_softmax_target,NULL);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,false,&attention_softmax_target,NULL);
			
			//cout<<"test source: "<<source_hidden_layers[i].upper_layer.source_side<<endl;
			//cout<<"test taget: "<<target_hidden_layers[i].upper_layer.source_side<<endl;

			attention_softmax_target.lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i]); //softmax	
		}
		else {
			source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
			target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[i+1]);
		}
	}
	
		
}


vector<int> Decoder::w2id(string input_sen)
{
    stringstream ss;
    ss << input_sen;
    string w;
    vector<int> wids;
    while (ss>>w)
    {
        if (src_w2i.find(w) != src_w2i.end() && src_w2i[w] < source_vocab_size) //
            wids.push_back(src_w2i[w]);
        else
            wids.push_back(0); //<UNK>
    }
    //wids.push_back(); no eos
    return wids;
}


string Decoder::id2w(vector<int> output_wids)
{
    string output_sen;
    for (int i=0;i<output_wids.size() - 1;i++)
    {
        output_sen += tgt_i2w[output_wids[i]] + " ";
    }
    return output_sen;
}


void Decoder::encode(vector<int> input_wids){
	
	B = 1;	
	T = input_wids.size();

	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

	
	input_layer_source.prepare_forward(&input_wids[T-1],input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
	input_layer_source.forward_prop(0, T, B);
	
	//for bidirection
	input_layer_source_bi.prepare_forward(&input_wids[0],input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
	input_layer_source_bi.forward_prop(0, T, B);
	
	for (int i=1; i<T; i++)
    {	
		input_layer_source_bi.prepare_forward(&input_wids[i],input_layer_source_bi.d_h_t,input_layer_source_bi.d_c_t);
		input_layer_source_bi.forward_prop(i, T, B);
	}
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	

	for(int j=0; j<source_hidden_layers.size(); j++){
		source_hidden_layers[j].prepare_forward(source_hidden_layers[j].d_init_hidden_vector,source_hidden_layers[j].d_init_cell_vector);
		source_hidden_layers[j].forward_prop(0, T, B);		
	}	
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

	for (int i=1; i<T; i++)
    {	
		input_layer_source.prepare_forward(&input_wids[T-1-i],input_layer_source.d_h_t,input_layer_source.d_c_t);
		input_layer_source.forward_prop(i, T, B);
		for(int j=0; j<source_hidden_layers.size(); j++){
			source_hidden_layers[j].prepare_forward(source_hidden_layers[j].d_h_t,source_hidden_layers[j].d_c_t);
			source_hidden_layers[j].forward_prop(i, T, B);		
		}	
		
		cudaSetDevice(gpu_num);
		cudaDeviceSynchronize();
	
	}

	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();

}

void Decoder::get_next_prob(vector<int> tgt_wids, int index) {
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	
	if(index == 0) {
		input_layer_target.prepare_forward(&tgt_wids[0],input_layer_source.d_h_t,input_layer_source.d_c_t);
		input_layer_target.forward_prop(0, T, B);
		for(int j=0; j<target_hidden_layers.size(); j++) {
			target_hidden_layers[j].prepare_forward(source_hidden_layers[j].d_h_t,source_hidden_layers[j].d_c_t);
			target_hidden_layers[j].forward_prop(0, T, B);
		}
	}
	else {
		
		input_layer_target.prepare_forward_decode(&tgt_wids[0], &father_idx[0], B, attention_softmax_target.d_final_temp_2, attention_softmax_target.gpu_info.e0); //
		input_layer_target.forward_prop(index, T, B);
		for(int j=0; j<target_hidden_layers.size(); j++) {
			target_hidden_layers[j].prepare_forward_decode(&father_idx[0], B, attention_softmax_target.gpu_info.e0); //
			target_hidden_layers[j].forward_prop(index, T, B);
		}
		
	}	

	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
	
	attention_softmax_target.attention_softmax_forward_prop(T,B); // T: source length, B: current beam size
	
	cudaSetDevice(gpu_num);
	cudaDeviceSynchronize();
}

void Decoder::generate_new_samples(vector<vector<int>> &hyp_samples, vector<float> &hyp_scores,
		vector<vector<int>> &final_samples, vector<float> &final_scores, int &dead_k, vector<int> &tgt_wids, vector<int> &father_idx) {
	
	int live_k = beam_size - dead_k;
	vector<float> logit_softmax;
	logit_softmax.resize(target_vocab_size*beam_size);
	cudaMemcpy(&logit_softmax[0], attention_softmax_target.d_logit_softmax, target_vocab_size*B*sizeof(float), cudaMemcpyDeviceToHost);
	priority_queue<pair<float,pair<int,int> >,vector<pair<float,pair<int,int> > >,greater<pair<float,pair<int,int> > > > q;

	for (int i=0;i<B;i++){
		for (int j=0;j<target_vocab_size;j++) {
			float score = log(logit_softmax[IDX2C(j,i,target_vocab_size)]) + hyp_scores[i]; // (target_vocab_size,B)
			if (q.size() < live_k )
				q.push(make_pair(score, make_pair(i,j)));
			else {
				if (q.top().first < score) {
					q.pop();  // discard small
					q.push(make_pair(score, make_pair(i,j)));
				}
			}

		}

	}

	vector<vector<int>> new_hyp_samples;
	vector<float> new_hyp_scores;
	//vector<int> father_idx;
	father_idx.clear();
	tgt_wids.clear();

	for(int k=0; k<live_k; k++) {
	
		float score = q.top().first; // q small -> big

		int i = q.top().second.first;
		int j = q.top().second.second;
		vector<int> sample(hyp_samples[i]); //
		sample.push_back(j);
		if(j==1) {
			dead_k += 1;
			final_samples.push_back(sample);
			float lp = pow((5+sample.size()),0.2)/pow(6,0.2);
			float score_my = score/lp;
			final_scores.push_back(score_my);
		}
		else {
			new_hyp_samples.push_back(sample);
			new_hyp_scores.push_back(score);
			tgt_wids.push_back(j);
			father_idx.push_back(i);
		}
		q.pop();
	}

	hyp_samples.swap(new_hyp_samples);
	hyp_scores.swap(new_hyp_scores);
}

vector<int> Decoder::decode() {
	
	//
	vector<vector<int>> final_samples;
	vector<float> final_scores;
	vector<vector<int>> hyp_samples(1,vector<int>());
	vector<float> hyp_scores(1,0.0);
	int dead_k = 0;

	vector<int> tgt_wids;
	tgt_wids.push_back(0);

	for(int i=0; i<Tmax; i++) {
		get_next_prob(tgt_wids, i);
		generate_new_samples(hyp_samples, hyp_scores, final_samples, final_scores, dead_k, tgt_wids, father_idx);
		//cin>>a;

		B = beam_size-dead_k;
		if(B<=0) {
			break;
		}
	}
	if(B>0) {
		for(int k=0; k<B; k++) {
			final_samples.push_back(hyp_samples[k]);
			final_scores.push_back(hyp_scores[k]);
		}
	}

	float best_score = -9999;
	int best_k = 0;
	for(int k=0; k<final_samples.size(); k++) {
		float score = final_scores[k]/final_samples[k].size();
		if(score>best_score) {
			best_score = score;
			best_k = k;
		}
	}
	return final_samples[best_k];
}


string Decoder::translate(string input_sen)
{
	vector<int> input_wids = w2id(input_sen);
	
	//cout<<"******************start source forward prop*******************"<<endl;
	encode(input_wids);
	
	//cout<<"******************start target forward prop*******************"<<endl;
    vector<int> output_wids = decode();
    
	string output_sen = id2w(output_wids);
    return output_sen;
}


