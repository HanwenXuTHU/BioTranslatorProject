import os
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from .BioUtils import extract_terms_from_dataset
from .BioUtils import load, load_obj, proteinData, emb2tensor, gen_go_emb, organize_workingspace, term_training_numbers


class BioLoader:
    '''
    This class loads and stores the data we used in BioTranslator
    '''
    def __init__(self, cfg):
        # Organize the working space
        organize_workingspace(cfg.working_space)
        # load gene ontology data
        self.go_data = load(cfg.go_file)
        # load mappings between terms and index, also find the intersection with go terms
        self.load_terms(cfg)
        # load proteins which need to be excluded
        self.load_eval_uniprots(cfg)
        # load dataset
        self.load_dataset(cfg)
        # generate proteinData, which is defined in BioUtils
        self.gen_protein_data(cfg)
        # load textual description embeddings of go terms
        self.load_text_emb(cfg)
        print('Data Loading Finished!')

    def load_terms(self, cfg):
        train_terms = pd.read_pickle(cfg.train_terms_file)
        self.train_i2terms = list(train_terms['terms'])
        go_terms = list(self.go_data.keys())
        self.train_i2terms = list(set(self.train_i2terms).intersection(set(go_terms)))
        self.train_terms2i = collections.OrderedDict()
        self.n_classes = len(self.train_i2terms)
        print('train terms number:{}'.format(len(self.train_i2terms)))
        for i in range(len(self.train_i2terms)): self.train_terms2i[self.train_i2terms[i]] = i

    def load_eval_uniprots(self, cfg):
        eval_dataset = cfg.excludes
        eval_uniprots = []
        for d in eval_dataset:
            data_i = load_obj(cfg.data_repo + d + '/pathway_dataset.pkl')
            eval_uniprots += list(data_i['proteins'])
        self.eval_uniprots = set(eval_uniprots)

    def load_dataset(self, cfg):
        # load train dataset
        self.train_data = pd.read_pickle(cfg.train_file)
        # exclude proteins in pathway dataset from the train data
        drop_index = []
        for i in self.train_data.index:
            if self.train_data.loc[i]['proteins'] in self.eval_uniprots:
                drop_index.append(i)
        self.train_data = self.train_data.drop(index=drop_index)
        # load protein network fatures and description features
        self.train_prot_network = load_obj(cfg.train_prot_network_file)
        self.network_dim = np.size(list(self.train_prot_network.values())[0])
        self.train_prot_description = load_obj(cfg.train_prot_description_file)
        # load eval dataset (pathway dataset)
        self.eval_data = pd.read_pickle(cfg.eval_file)
        self.eval_prot_network = load_obj(cfg.eval_prot_network_file)
        self.eval_prot_description = load_obj(cfg.eval_prot_description_file)
        # load eval terms
        self.eval_terms2i, self.eval_i2terms = extract_terms_from_dataset(self.eval_data)
        print('eval pathway number:{}'.format(len(self.eval_i2terms)))

    def gen_protein_data(self, cfg):
        # generate protein data which can be loaded by torch
        # the raw train and test data is for blast preds function in BioTrainer
        self.raw_train, self.raw_val = self.train_data.copy(deep=True), self.eval_data.copy(deep=True)

        self.train_data = proteinData(self.train_data, self.train_terms2i, self.train_prot_network,
                                                 self.train_prot_description, gpu_ids=cfg.gpu_ids)
        self.eval_data = proteinData(self.eval_data, self.eval_terms2i, self.eval_prot_network,
                                                   self.eval_prot_description, gpu_ids=cfg.gpu_ids)

    def load_text_emb(self, cfg):
        if not os.path.exists(cfg.emb_path):
            os.mkdir(cfg.emb_path)
            print('Warning: We created the embedding folder: {}'.format(cfg.emb_path))
        if cfg.emb_name not in os.listdir(cfg.emb_path):
            gen_go_emb(cfg)
        cfg.text_embedding_file = cfg.emb_path + cfg.emb_name
        self.text_embeddings = pd.read_pickle(cfg.text_embedding_file)
        self.text_embeddings = emb2tensor(self.text_embeddings, self.train_terms2i)

        self.pathway_embeddings = pd.read_pickle(cfg.pathway_emb_file)
        self.pathway_embeddings = emb2tensor(self.pathway_embeddings, self.eval_terms2i)
        if len(cfg.gpu_ids) > 0:
            self.text_embeddings = self.text_embeddings.float().cuda()
            self.pathway_embeddings = self.pathway_embeddings.float().cuda()