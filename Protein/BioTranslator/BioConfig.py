class BioConfig:
    def __init__(self, args: dict):
        # load args
        self.load_args(args)
        # load dir of the dataset
        data_dir = self.data_repo + self.dataset + '/'
        # load dataset file
        self.go_file, self.train_fold_file, self.validation_fold_file, \
        self.terms_file, self.zero_shot_term_path, self.prot_network_file, \
        self.prot_description_file = self.read_dataset_file(data_dir)
        # generate other parameters
        self.gen_other_parameters()

    def load_args(self, args):
        # load the settings in args
        self.dataset = args['dataset'].strip()
        self.method = args['method'].strip()
        self.task = args['task'].strip()
        self.data_repo = args['data_repo'].strip()
        self.encoder_path = args['encoder_path'].strip()
        self.emb_path = args['emb_path'].strip()
        self.working_space = args['working_space'].strip()
        self.save_path = args['save_path'].strip()
        self.max_length = args['max_length']
        self.hidden_dim = args['hidden_dim']
        self.features = args['features'].split(', ')
        self.lr = args['lr']
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.gpu_ids = args['gpu_ids'].strip()

    def read_dataset_file(self, data_dir: str):
        go_file = data_dir + 'go.obo'
        train_fold_file = data_dir + 'train_data_fold_{}.pkl'
        validation_fold_file = data_dir + 'validation_data_fold_{}.pkl'
        terms_file = data_dir + 'terms.pkl'
        zero_shot_term_path = data_dir + 'zero_shot_terms_fold_{}.pkl'
        prot_network_file = data_dir + 'prot_network.pkl'
        prot_description_file = data_dir + 'prot_description.pkl'
        return go_file, train_fold_file, validation_fold_file, terms_file, zero_shot_term_path, prot_network_file, prot_description_file

    def gen_other_parameters(self):
        # Other paramters
        # k-fold cross-validation
        self.k_fold = 3
        # number of ammino acids of protein sequences
        self.seq_input_nc = 21
        # number of channels in the CNN architecture
        self.seq_in_nc = 512
        # the max size of CNN kernels
        self.seq_max_kernels = 129
        # the dimension of term text/graph embeddings
        if self.method in ['BioTranslator', 'ProTranslator']:
            self.term_enc_dim = 768
        elif self.method == 'clusDCA':
            self.term_enc_dim = 500
        else:
            self.term_enc_dim = 1000
        # load the Diamond score related results
        if self.task == 'few_shot':
            self.diamond_score_path = self.data_repo + self.dataset + '/validation_data_fold_{}.res'
            self.blast_preds_path = self.data_repo + self.dataset + '/blast_preds_fold_{}.pkl'
            # the alhpa paramter we used in DeepGOPlus
            self.ont_term_syn = {'biological_process': 'bp', 'molecular_function': 'mf', 'cellular_component': 'cc'}
            self.alphas = {"mf": 0.68, "bp": 0.63, "cc": 0.46}
            self.blast_res_name = self.working_space + '{}/'.format(self.task) + \
                                'results/{}_{}_blast.pkl'.format(self.method, self.dataset)
        # where you store the deep learning model
        self.save_model_path = self.working_space + '{}/'.format(self.task) + 'model/{}_{}_{}.pth'
        # the name of logger file, that contains the information of training process
        self.logger_name = self.working_space + '{}/'.format(self.task) + 'log/{}_{}.log'.format(self.method, self.dataset)
        # where you save the results of BioTranslator
        self.results_name = self.working_space + '{}/'.format(self.task) + 'results/{}_{}.pkl'.format(self.method, self.dataset)
        # get the name of textual description embeddings
        self.emb_name = '{}_go_embeddings.pkl'.format(self.method)