class data_loading:

    def __init__(self):
        self.dataset = 'GOA_Human_Pathway'  # choose from [GOA_Human, GOA_Yeast, GOA_Mouse, SwissProt, CAFA3]
        self.data_dir = '../data/'  # where you put the dataset
        self.pathway_dir = '../data/'
        self.def_embedding_file = '/home/hwxu/BioTranslator/Protein_Embeddings/embeddings/PubMedFull_go_embeddings.pkl'  # where you put the textual embeddings
        self.gpu_ids = '0'
        self.emb_options = 'PubMedFull'
        self.logger_name = '_{}'.format(self.emb_options)
        self.generate_parameters()

    def generate_parameters(self):
        root_file = self.data_dir + self.dataset + '/'
        self.go_file = root_file + 'go.obo'
        self.train_data_file = root_file + 'train_data.pkl'
        self.terms_file = root_file + 'terms.pkl'
        self.prot_vector_file = root_file + 'prot_vector.pkl'
        self.prot_description_file = root_file + 'prot_description.pkl'
        self.text_mode = 'def'


class model_config:

    def __init__(self):
        self.lr = 0.0003
        self.input_nc = 21
        self.in_nc = 512
        self.n_classes = 5101
        self.max_kernels = 129
        self.hidden_dim = [1500]
        self.features = ['seqs', 'protein description']
        self.max_len = 2000
        self.N_vector_dim = 500
        self.batch_size = 8
        self.epoch = 30
        self.gpu_ids = '0'
        self.dropout = 0
        self.save_path = '/home/hwxu/BioTranslator/Protein_Pathway_Text2Graph/results/model/{}_model.pth'


class Text2Graph_config:

    def __init__(self):
        self.lr = 0.0005
        self.input_nc = 4
        self.seqL = 2000
        self.x_dim = 768
        self.graph_emb_dim = 768
        self.num_layers = 2
        self.GAT_heads = 4
        self.gpu_ids = '0'
        self.epoch = 1
        self.seen_ratio = 0.1
        self.test_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.with_graph = False
        self.with_text = False
        self.save_auroc = 'results/graph_text_{}_{}'.format(self.with_graph, self.with_text)
