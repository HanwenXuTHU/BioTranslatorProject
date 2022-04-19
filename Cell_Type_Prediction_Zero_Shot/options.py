class data_loading_opt:

    def __init__(self):
        self.dname = 'muris_droplet' #you need to specify the dataset name, choose from [Tabula_Spaiens, Tabula_Microcebus, 'muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
        self.scrna_data_dir = '/home/hwxu/sc_data/OnClass_data_public/scRNA_data/' #where you put the dataset
        self.ontology_data_dir = '/home/hwxu/cellZSL/data/Ontology_data/' #where you save the ontology data
        self.backup_file = 'cache/sparse_featurefile_backup_raw.h5ad' #where you save the sparse features
        self.memory_saving_mode = True
        self.generate_parameters()

    def generate_parameters(self):
        self.train_label = 'cell_ontology_id'
        self.test_label = 'cell_ontology_id'
        print('Data name :{}'.format(self.dname))
        self.n_iter = 5
        self.unseen_ratio = [0.9, 0.1, 0.7, 0.5, 0.3]
        self.nfold_sample = 0.2
        self.is_gpu = True
        self.text_mode = 'def'


class model_config:
    def __init__(self):
        self.data_opt = data_loading_opt()
        self.batch_size = 128
        self.lr = 0.0001
        self.nhidden = [30]
        self.epoch = 15
        self.drop_out = 0.05
        self.save_id = 'results/model/model_'# save the model file
        self.results_path = 'results/inference/save_results.pkl' #save the results file
        self.fig_path = 'results/inference/save_fig.pdf' #save the results fig
