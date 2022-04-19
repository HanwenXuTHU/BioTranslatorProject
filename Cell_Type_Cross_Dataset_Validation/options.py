class data_loading_opt:

    def __init__(self):
        self.scrna_data_dir = '/home/hwxu/sc_data/OnClass_data_public/scRNA_data/'#where you put the dataset
        self.ontology_data_dir = '/home/hwxu/cellZSL/data/Ontology_data/'#where you save the ontology data
        self.backup_file1 = 'cache/sparse_featurefile_backup_raw3.h5ad'
        self.backup_file2 = 'cache/sparse_featurefile_backup_raw4.h5ad'
        self.memory_saving_mode = True
        self.generate_parameters()

    def generate_parameters(self):
        self.train_label = 'cell_ontology_id'
        self.test_label = 'cell_ontology_id'
        self.n_iter = 5
        self.unseen_ratio = [0.9, 0.1, 0.7, 0.5, 0.3]
        self.nfold_sample = 0.2
        self.is_gpu = True
        self.text_mode = 'def'
        self.train_dname = ''
        self.eval_dnames = ['Tabula_Sapiens', 'Tabula_Microcebus', 'muris_droplet',
                    'microcebusAntoine', 'microcebusBernard', 'microcebusMartine',
                    'microcebusStumpy', 'muris_facs']


class model_config:
    def __init__(self):
        self.data_opt = data_loading_opt()
        self.batch_size = 128
        self.lr = 0.0001
        self.nhidden = [30]
        self.epoch = 15
        self.drop_out = 0.05
        self.save_id = 'results/model/{}_{}'#where you save the model
        self.results_path = 'results/inference/{}_{}_results.pkl'#where you save the results
        self.save_fig = 'results/inference/cross_validation_{}.pdf'
