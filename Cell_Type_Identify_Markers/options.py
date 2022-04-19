class data_loading_opt:

    def __init__(self):
        self.dname = 'muris_droplet' #you need to specify the dataset name, choose from [Tabula_Spaiens, Tabula_Microcebus, 'muris_droplet','microcebusAntoine','microcebusBernard','microcebusMartine','microcebusStumpy','muris_facs']
        self.scrna_data_dir = '/home/hwxu/sc_data/OnClass_data_public/scRNA_data/'
        self.train_label = 'cell_ontology_id'
        self.test_label = 'cell_ontology_id'
        self.ontology_data_dir = '/home/hwxu/cellZSL/data/Ontology_data/'
        self.backup_file = 'cache/sparse_featurefile_backup_raw.h5ad'
        self.memory_saving_mode = True
        self.n_iter = 1
        self.unseen_ratio = [0.5]
        self.nfold_sample = 0.2
        self.is_gpu = True
        self.text_mode = 'def'
        print('Data name :{}'.format(self.dname))


class model_config:
    def __init__(self):
        self.data_opt = data_loading_opt()
        self.batch_size = 128
        self.lr = 0.0001
        self.nhidden = [30]
        self.epoch = 15
        self.drop_out = 0.05
        self.save_id = 'results/model/{}_{}_'.format(self.data_opt.dname, self.data_opt.text_mode)
        self.save_marker_path = 'results/markers/{}_{}_markers.pkl'.format(self.data_opt.dname, self.data_opt.text_mode)
