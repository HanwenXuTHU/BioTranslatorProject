class BioConfig:
    def __init__(self, args: dict):
        # load args
        self.load_args(args)
        # load dir of the dataset
        train_data_dir = self.data_repo + self.dataset + '/'
        # load dataset file
        self.go_file, self.train_file, _, _, \
        self.train_terms_file, _, self.train_prot_network_file, \
        self.train_prot_description_file = self.read_dataset_file(train_data_dir)
        # load pathway dataset file
        eval_data_dir = self.data_repo + self.pathway_dataset + '/'
        _, _, _, _, \
        self.eval_terms_file, _, self.eval_prot_network_file, \
        self.eval_prot_description_file = self.read_dataset_file(eval_data_dir)
        self.eval_file = eval_data_dir + 'pathway_dataset.pkl'
        # generate other parameters
        self.gen_other_parameters()

    def load_args(self, args):
        # load the settings in args
        self.dataset = args['dataset'].strip()
        self.pathway_dataset = args['pathway_dataset'].strip()
        self.excludes = args['excludes']
        self.method = args['method'].strip()
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
        train_file = data_dir + 'dataset.pkl'
        train_fold_file = data_dir + 'train_data_fold_{}.pkl'
        validation_fold_file = data_dir + 'validation_data_fold_{}.pkl'
        terms_file = data_dir + 'terms.pkl'
        zero_shot_term_path = data_dir + 'zero_shot_terms_fold_{}.pkl'
        prot_network_file = data_dir + 'prot_network.pkl'
        prot_description_file = data_dir + 'prot_description.pkl'
        return go_file, train_file, train_fold_file, validation_fold_file, terms_file, zero_shot_term_path, prot_network_file, prot_description_file

    def gen_other_parameters(self):
        # Other paramters
        # select the nearest k GO term embeddings when annotate the pathway
        self.nearest_k = 5
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
        # where you store the deep learning model
        self.save_model_path = self.working_space + 'model/{}_{}.pth'
        # the name of logger file, that contains the information of training process
        self.logger_name = self.working_space + 'log/{}_{}.log'.format(self.method, self.pathway_dataset)
        # where you save the results of BioTranslator
        self.results_name = self.working_space + 'results/{}_{}.pkl'.format(self.method, self.pathway_dataset)
        # get the name of train data textual description embeddings
        self.emb_name = '{}_go_embeddings.pkl'.format(self.method)
        # get the path of pathway textual description embeddings
        self.pathway_emb_file = self.data_repo + self.pathway_dataset + '/pathway_embeddings.pkl'