class data_loading:

    def __init__(self):
        '''
        Here you can specify parameters related to loading your dataset
        '''
        self.dataset = 'GOA_Human'# choose from [GOA_Human, GOA_Yeast, GOA_Mouse, SwissProt, CAFA3]
        self.data_dir = '../data/' # where you put the dataset
        self.def_embedding_file = '/home/hwxu/BioTranslator/Protein_Embeddings/embeddings/PubMedFull_go_embeddings.pkl'# where you put the textual embeddings
        self.logger_name = 'results/training_log/save_log.log' # where you save the training log
        self.gpu_ids = '1'
        self.generate_parameters()

    def generate_parameters(self):
        root_file = self.data_dir + self.dataset + '/'
        self.go_file = root_file + 'go.obo'
        self.train_fold_file = root_file + 'train_data_fold_{}.pkl'
        self.validation_fold_file = root_file + 'validation_data_fold_{}.pkl'
        self.terms_file = root_file + 'terms.pkl'
        self.zero_shot_term_path = root_file + 'zero_shot_terms_fold_{}.pkl'
        self.prot_vector_file = root_file + 'prot_vector.pkl'
        self.prot_description_file = root_file + 'prot_description.pkl'
        self.text_mode = 'def'
        self.k_fold = 3


class model_config:

    def __init__(self):
        '''
        Here you can setup the paramters related to the model training and saving
        '''
        self.lr = 0.0003 # learning rate
        self.input_nc = 21 # the number of one hot encodings
        self.in_nc = 512 # the number of input channels
        self.max_kernels = 129 # The largest size of convolution kernel
        self.hidden_dim = [1500] # the hidden dimension
        self.features = ['seqs', 'protein description', 'network'] #The features you chose to use when predicting protein functions
        self.max_len = 2000 # the max length of protein sequences
        self.batch_size = 32
        self.epoch = 30
        self.gpu_ids = '1'
        self.save_path = 'results/cache/model_{}.pth' # where you store the models


class inference_config:

    def __init__(self):
        '''
        Here you can specify where you choose to save the performance on the test set
        '''
        self.save_auroc_path = 'results/inference/auroc.pkl'
        self.save_auroc_geq_threshold_percentage = 'results/inference/auroc_geq_threshold_percentage.pkl'
        self.barplot_save_path = 'results/inference/auroc.pdf'
        self.auroc_geq_threshold_barplot_save_path = 'results/inference/auroc_geq_T.pdf'