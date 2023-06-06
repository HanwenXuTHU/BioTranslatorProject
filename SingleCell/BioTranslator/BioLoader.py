import os
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from .BioUtils import gen_co_emb, emb2tensor, cellData, SplitTrainTest, find_gene_ind
from .BioUtils import organize_workingspace, DataProcessing, read_data, read_data_file, read_ontology_file


class BioLoader:
    '''
    This class loads and stores the data we used in BioTranslator
    '''
    def __init__(self, cfg):
        # Organize the working space
        organize_workingspace(cfg.working_space, cfg.task)
        # load textual description embeddings of cell ontology terms
        self.load_text_emb(cfg)
        # load files
        self.load_files(cfg)
        print('Data Loading Finished!')

    def load_files(self, cfg):
        self.cfg = cfg
        self.terms_with_def = self.text_embeddings.keys()
        self.cell_type_nlp_emb_file, self.cell_type_network_file, self.cl_obo_file = read_ontology_file('cell ontology', cfg.ontology_repo)
        self.DataProcess_obj = DataProcessing(self.cell_type_network_file, self.cell_type_nlp_emb_file, memory_saving_mode=cfg.memory_saving_mode, terms_with_def=self.terms_with_def)
        feature_file, filter_key, label_key, label_file, gene_file = read_data_file(cfg.dataset, cfg.data_repo)
        self.feature, self.genes, self.label, _, _ = read_data(feature_file, cell_ontology_ids=self.DataProcess_obj.cell_ontology_ids,
                                                exclude_non_leaf_ontology=True, tissue_key='tissue',
                                                filter_key=filter_key, AnnData_label_key=label_key,
                                                nlp_mapping=False, cl_obo_file=self.cl_obo_file,
                                                cell_ontology_file=self.cell_type_network_file, co2emb=self.DataProcess_obj.co2vec_nlp,
                                                               memory_saving_mode=cfg.memory_saving_mode,
                                                               backup_file=cfg.backup_file)

    def load_text_emb(self, cfg):
        if not os.path.exists(cfg.emb_path):
            os.mkdir(cfg.emb_path)
            print('Warning: We created the embedding folder: {}'.format(cfg.emb_path))
        if cfg.emb_name not in os.listdir(cfg.emb_path):
            gen_co_emb(cfg)
        cfg.text_embedding_file = cfg.emb_path + cfg.emb_name
        self.text_embeddings = pd.read_pickle(cfg.text_embedding_file)

    def read_eval_files(self, cfg, eval_dname):
        feature_file, filter_key, label_key, label_file, gene_file = read_data_file(eval_dname, cfg.data_repo)
        self.eval_feature, self.eval_genes, self.eval_label, _, _ = read_data(feature_file, cell_ontology_ids=self.DataProcess_obj.cell_ontology_ids,
                                                                                 exclude_non_leaf_ontology=True, tissue_key='tissue',
                                                                                 filter_key=filter_key, AnnData_label_key=label_key,
                                                                                 nlp_mapping=False, cl_obo_file=self.cl_obo_file,
                                                                                 cell_ontology_file=self.cell_type_network_file,
                                                                                 co2emb=self.DataProcess_obj.co2vec_nlp, memory_saving_mode=self.cfg.memory_saving_mode,
                                                                                 backup_file=self.cfg.eval_backup_file)

    def generate_data(self, iter=None, unseen_ratio=None):
        if self.cfg.task == 'same_dataset':
            train_feature, train_label, test_feature, test_label = SplitTrainTest(self.feature,
                                                                                  self.label,
                                                                                  nfold_cls=unseen_ratio,
                                                                                  nfold_sample=self.cfg.nfold_sample,
                                                                                  random_state=iter,
                                                                                  memory_saving_mode=self.cfg.memory_saving_mode)
            train_genes, test_genes = self.genes, self.genes
            self.DataProcess_obj.EmbedCellTypes(train_label)
            train_feature, test_feature, train_genes, test_genes = self.DataProcess_obj.ProcessTrainFeature(
                train_feature, train_label, train_genes, test_feature=test_feature, test_genes=test_genes,
                batch_correct=False)

            test_feature, mapping = self.DataProcess_obj.ProcessTestFeature(test_feature, test_genes, log_transform=False)

            self.unseen2i = collections.OrderedDict()
            self.train2i = collections.OrderedDict()
            self.test2i = collections.OrderedDict()
            idx = 0
            for x in set(train_label):
                self.train2i[x] = idx
                idx += 1

            idx = 0
            for x in set(test_label).union(set(train_label)):
                self.test2i[x] = idx
                if x not in set(train_label):
                    self.unseen2i[x] = idx
                idx += 1

            train_label = [self.train2i[x] for x in train_label]
            test_label = [self.test2i[x] for x in test_label]

            self.train_data = cellData(train_feature, train_label, nlabel=len(self.train2i.keys()), gpu_ids=self.cfg.gpu_ids)
            self.test_data = cellData(test_feature, test_label, nlabel=len(self.test2i.keys()), gpu_ids=self.cfg.gpu_ids)

            self.train_emb = emb2tensor(self.text_embeddings, self.train2i)
            if len(self.cfg.gpu_ids) > 0:
                self.train_emb = self.train_emb.cuda()
            self.test_emb = emb2tensor(self.text_embeddings, self.test2i)
            if len(self.cfg.gpu_ids) > 0:
                self.test_emb = self.test_emb.cuda()

            self.train_genes = train_genes
            self.test_genes = test_genes
            self.ngene = len(train_genes)
            self.ndim = self.train_emb.size(1)

        elif self.cfg.task == 'cross_dataset':
            self.read_eval_files(self.cfg, self.cfg.eval_dataset)
            common_genes = np.sort(list(set(self.genes) & set(self.eval_genes)))
            gid1 = find_gene_ind(self.genes, common_genes)
            gid2 = find_gene_ind(self.eval_genes, common_genes)
            train_feature = self.feature[:, gid1]
            eval_feature = self.eval_feature[:, gid2]
            train_genes = common_genes
            eval_genes = common_genes
            self.DataProcess_obj.EmbedCellTypes(self.label)
            train_feature, eval_feature, train_genes, test_genes = self.DataProcess_obj.ProcessTrainFeature(
                train_feature, self.label, train_genes, test_feature=eval_feature, test_genes=eval_genes,
                batch_correct=False)

            if self.cfg.memory_saving_mode:
                eval_feature, mapping = self.DataProcess_obj.ProcessTestFeature(eval_feature, eval_genes,
                                                                                log_transform=False)
                train_feature = train_feature.tocsr()
                eval_feature = eval_feature.tocsr()
            else:
                eval_feature = self.DataProcess_obj.ProcessTestFeature(eval_feature, eval_genes,
                                                                       log_transform=False)
            self.unseen2i = collections.OrderedDict()
            self.train2i = collections.OrderedDict()
            self.test2i = collections.OrderedDict()
            idx = 0
            for x in set(self.label):
                self.train2i[x] = idx
                idx += 1
            idx = 0
            for x in set(self.eval_label):
                self.test2i[x] = idx
                if x not in self.train2i.keys():
                    self.unseen2i[x] = idx
                idx += 1
            train_label = [self.train2i[x] for x in self.label]
            test_label = [self.test2i[x] for x in self.eval_label]
            self.train_data = cellData(train_feature, train_label, nlabel=len(self.train2i.keys()), gpu_ids=self.cfg.gpu_ids)
            self.test_data = cellData(eval_feature, test_label, nlabel=len(self.test2i.keys()), gpu_ids=self.cfg.gpu_ids)
            self.train_emb = emb2tensor(self.text_embeddings, self.train2i)
            if len(self.cfg.gpu_ids) > 0:
                self.train_emb = self.train_emb.cuda()
            self.test_emb = emb2tensor(self.text_embeddings, self.test2i)
            if len(self.cfg.gpu_ids) > 0:
                self.test_emb = self.test_emb.cuda()
            self.ngene = len(train_genes)
            self.ndim = self.train_emb.size(1)

