class BioConfig:
    def __init__(self, args: dict):
        # load args
        self.load_args(args)
        # generate other parameters
        self.gen_other_parameters()

    def load_args(self, args):
        # load the settings in args
        self.dataset = args['dataset'].strip()
        self.eval_dataset = args['eval_dataset'].strip()
        self.method = args['method'].strip()
        self.task = args['task'].strip()
        self.data_repo = args['data_repo'].strip()
        self.ontology_repo = args['ontology_repo'].strip()
        self.encoder_path = args['encoder_path'].strip()
        self.emb_path = args['emb_path'].strip()
        self.working_space = args['working_space'].strip()
        self.save_path = args['save_path'].strip()
        self.hidden_dim = args['hidden_dim']
        self.lr = args['lr']
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.gpu_ids = args['gpu_ids'].strip()

    def gen_other_parameters(self):
        # Other paramters
        # use expression as the features
        self.features = ['expression']
        # k-fold cross-validation
        self.n_iter = 5
        self.unseen_ratio = [0.9, 0.7, 0.5, 0.3, 0.1]
        self.nfold_sample = 0.2
        # the dropout
        self.drop_out = 0.05
        # the dimension of term text/graph embeddings
        self.term_enc_dim = 768
        # set the memory saving mode to True
        self.memory_saving_mode = True
        # where you store the backup files
        self.backup_file = self.working_space + '{}/'.format(self.task) + 'cache/sparse_backup_raw.h5ad'
        # where you store the deep learning model
        self.save_model_path = self.working_space + '{}/'.format(self.task) + 'model/{}_{}_{}_{}.pth'
        # the name of logger file, that contains the information of training process
        self.logger_name = self.working_space + '{}/'.format(self.task) + 'log/{}_{}.log'.format(self.method, self.dataset)
        # where you save the results of BioTranslator
        self.results_name = self.working_space + '{}/'.format(self.task) + 'results/{}_{}.pkl'.format(self.method, self.dataset)
        # get the name of textual description embeddings
        self.emb_name = '{}_co_embeddings.pkl'.format(self.method)
        # when the task is cross_dataset
        if self.task == 'cross_dataset':
            self.n_iter = 1
            self.unseen_ratio = ['cross_dataset']
            self.save_model_path = self.working_space + '{}/'.format(self.task) + 'model/{}_{}_{}.pth'
            self.logger_name = self.working_space + '{}/'.format(self.task) + 'log/{}_{}_{}.log'.format(self.method, self.dataset, self.eval_dataset)
            self.eval_backup_file = self.working_space + '{}/'.format(self.task) + 'cache/sparse_backup_eval.h5ad'
            self.results_name = self.working_space + '{}/'.format(self.task) + 'results/{}_{}_{}.pkl'.format(self.method,
                                                                                                          self.dataset,
                                                                                                          self.eval_dataset)
