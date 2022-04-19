'''
MIT License

Copyright (c) 2021 sheng wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing import normalize
from scipy import stats
from utils import read_data, read_ontology_file, subset_cell_type_nlp_network, SplitTrainTest, read_data_file
from OnClass_utils import *
import warnings
from anndata import read_h5ad
from options import data_loading_opt
import pickle
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import collections


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class cellData(Dataset):

    def __init__(self, expression_matrix, labels, nlabel=0, is_gpu=True):
        self.expression_matrix = expression_matrix
        self.labels = labels
        self.nlabel = nlabel
        self.is_gpu = is_gpu

    def __getitem__(self, item):
        x = np.asarray(self.expression_matrix[item, :].todense()).squeeze()
        y = np.zeros(self.nlabel)
        y[self.labels[item]] = float(1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if self.is_gpu:
            x = x.cuda()
            y = y.cuda()
        return {'features': x, 'label': y}

    def __len__(self):
        return len(self.labels)

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def emb2tensor(def_embeddings, terms, add_bias=False):
    ann_id = list(terms.keys())
    embedding_array = np.zeros((len(ann_id), np.size(def_embeddings[ann_id[0]], 1)))

    for t_id in ann_id:
        t_def = def_embeddings[t_id].reshape([1, -1])
        t_def = t_def / np.sqrt(np.sum(np.power(t_def, 2), axis=1))
        embedding_array[terms[t_id], :] = t_def
    rank_e = np.linalg.matrix_rank(embedding_array)
    if add_bias:
        embedding_array = np.column_stack((np.eye(len(ann_id)), embedding_array))
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array.float()


class DataProcessing:
    def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology',
                 cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb',
                 terms_with_def=set(),
                 memory_saving_mode=False):
        """
        Borrowed from OnClass paper
        Initialize OnClass model with a given cell-type network and cell-type embedding file.
        Also, you may set the memory_saving_mode to True to get a model that uses less RAM.
        """
        self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
        self.cell_type_network_file = cell_type_network_file
        self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = read_cell_type_nlp_network(self.cell_type_nlp_emb_file, self.cell_type_network_file)
        self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = subset_cell_type_nlp_network(self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids, terms_with_def)
        self.mode = memory_saving_mode

    def ProcessTrainFeature(self, train_feature, train_label, train_genes, test_feature=None, test_genes=None,
                            batch_correct=False, log_transform=True):
        """
        Borrowed from OnClass paper
        Process the gene expression matrix used to train the model, and optionally the test data.
        Parameters
        ----------
        train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
            gene expression matrix of cell types
        train_label: `numpy.ndarray`
            labels for the training features
        train_genes: `list`
            list of genes used during training
        test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode), optional (None)
            gene expression matrix of cell types for the test set
        test_genes: `list`, optional (None)
            list of genes used in test set
        batch_correct: `bool`, optional (False)
            whether to correct for batch effect in data
        log_transform:`bool`, optional (True)
            whether to apply log transform to data
        Returns
        -------
        train_feature, test_feature, self.genes, self.genes
            returns the training feature gene expression matrix and the list of genese associated
            with it. If test_feature was not none, also returns the test features and their genes.
        """

        if log_transform is False and np.max(train_feature) > 1000:
            warnings.warn(
                "Max expression is" + str(np.max(train_feature)) + '. Consider setting log transform = True\n')
        self.genes = train_genes
        # batch correction is currently not supported for memory_saving_mode
        if batch_correct and test_feature is not None and test_genes is not None and self.mode:
            train_feature, test_feature, selected_train_genes = process_expression(train_feature, test_feature,
                                                                                   train_genes, test_genes)
            self.genes = selected_train_genes
        elif log_transform:
            if self.mode:
                train_feature = csr_matrix.log1p(train_feature)
            else:
                train_feature = np.log1p(train_feature)

            if test_feature is not None:
                if self.mode:
                    test_feature = csr_matrix.log1p(test_feature)
                else:
                    test_feature = np.log1p(test_feature)
        self.train_feature = train_feature
        self.train_label = train_label
        if test_feature is not None:
            return train_feature, test_feature, self.genes, self.genes
        else:
            return train_feature, self.genes

    def ProcessTestFeature(self, test_feature, test_genes, use_pretrain=None, batch_correct=False, log_transform=True):
        """
        Borrowed from OnClass paper
        Process the gene expression matrix used to test the model.
        Parameters
        ----------
        test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
            gene expression matrix of cell types for the test set
        test_genes: `list`
            list of genes used in test set
        use_pretrain: `string`, optional (None)
            name of the pretrained model
        batch_correct: `bool`, optional (False)
            whether to correct for batch effect in data
        log_transform:`bool`, optional (True)
            whether to apply log transform to data
        Returns
        -------
        gene_mapping or test_feature
            processes the test features and returns a data structure that encodes the gene
            expression matrix that should be used for testing. If the model is in memory saving
            mode, then the function will return a tuple of gene expression matrix and index array,
            otherwise, it will just return the matrix.
        """
        if log_transform is False and np.max(test_feature) > 1000:
            warnings.warn("Max expression is" + str(np.max(test_feature)) + '. Consider setting log transform = True\n')

        if log_transform:
            test_feature = np.log1p(test_feature)
        if batch_correct and not self.mode:
            test_feature = mean_normalization(self.train_feature_mean, test_feature)

        if self.mode:
            gene_mapping = get_gene_mapping(test_genes, self.genes)
            return test_feature, gene_mapping
        else:
            test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
            return test_feature

    def EmbedCellTypes(self, train_Y_str, dim=5, emb_method=3, use_pretrain=None, write2file=None):
        """
        Borrowed from OnClass paper
        Embed the cell ontology
        Parameters
        ----------
        cell_type_network_file : each line should be cell_type_1\tcell_type_2\tscore for weighted network or cell_type_1\tcell_type_2 for unweighted network
        dim: `int`, optional (500)
            Dimension of the cell type embeddings
        emb_method: `int`, optional (3)
            dimensionality reduction method
        use_pretrain: `string`, optional (None)
            use pretrain file. This should be the numpy file of cell type embeddings. It can read the one set in write2file parameter.
        write2file: `string`, optional (None)
            write the cell type embeddings to this file path
        Returns
        -------
        co2emb, co2i, i2co
            returns three dicts, cell type name to embeddings, cell type name to cell type id and cell type id to embeddings.
        """

        self.unseen_co, self.co2i, self.i2co, self.ontology_dict, self.ontology_mat = creat_cell_ontology_matrix(
            train_Y_str, self.co2co_graph, self.cell_ontology_ids, dfs_depth=3)
        self.nco = len(self.i2co)
        Y_emb = emb_ontology(self.i2co, self.ontology_mat, dim=dim, mi=emb_method, co2co_nlp=self.co2co_nlp,
                             unseen_l=self.unseen_co)
        self.co2emb = np.column_stack((np.eye(self.nco), Y_emb))
        self.nunseen = len(self.unseen_co)
        self.nseen = self.nco - self.nunseen
        self.co2vec_nlp_mat = np.zeros((self.nco, len(self.co2vec_nlp[self.i2co[0]])))
        for i in range(self.nco):
            self.co2vec_nlp_mat[i, :] = self.co2vec_nlp[self.i2co[i]]
        return self.co2emb, self.co2i, self.i2co, self.ontology_mat


class FileLoader:
    def __init__(self, opt):
        '''
        Borrowed from OnClass paper
        '''
        print('Text mode: {}'.format(opt.text_mode))
        self.cell_def_emb = load_obj(opt.ontology_data_dir + '/co_{}_pubmedbert.pkl'.format(opt.text_mode))
        self.terms_with_def = self.cell_def_emb.keys()
        self.cell_type_nlp_emb_file, self.cell_type_network_file, self.cl_obo_file = read_ontology_file('cell ontology', opt.ontology_data_dir)
        self.DataProcess_obj = DataProcessing(self.cell_type_network_file, self.cell_type_nlp_emb_file, memory_saving_mode=opt.memory_saving_mode, terms_with_def=self.terms_with_def)
        self.opt = opt
        feature_file, filter_key, label_key, label_file, gene_file = read_data_file(opt.dname, opt.scrna_data_dir)
        self.feature, self.genes, self.label, _, _ = read_data(feature_file, cell_ontology_ids=self.DataProcess_obj.cell_ontology_ids,
                                                exclude_non_leaf_ontology=True, tissue_key='tissue',
                                                filter_key=filter_key, AnnData_label_key=label_key,
                                                nlp_mapping=False, cl_obo_file=self.cl_obo_file,
                                                cell_ontology_file=self.cell_type_network_file, co2emb=self.DataProcess_obj.co2vec_nlp,
                                                               memory_saving_mode=opt.memory_saving_mode,
                                                               backup_file=opt.backup_file)

    def generate_data(self, iter, unseen_ratio):

        train_feature, train_label, test_feature, test_label = SplitTrainTest(self.feature, self.label, nfold_cls=unseen_ratio, nfold_sample=self.opt.nfold_sample,
                                                                      random_state=iter, memory_saving_mode=self.opt.memory_saving_mode)
        train_genes = self.genes
        test_genes = self.genes
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

        self.train_data = cellData(train_feature, train_label, nlabel=len(self.train2i.keys()), is_gpu=self.opt.is_gpu)
        self.test_data = cellData(test_feature, test_label, nlabel=len(self.test2i.keys()), is_gpu=self.opt.is_gpu)

        self.train_emb = emb2tensor(self.cell_def_emb, self.train2i)
        if self.opt.is_gpu:
            self.train_emb = self.train_emb.cuda()
        self.test_emb = emb2tensor(self.cell_def_emb, self.test2i)
        if self.opt.is_gpu:
            self.test_emb = self.test_emb.cuda()

        self.train_genes = train_genes
        self.test_genes = test_genes
        self.ngene = len(train_genes)
        self.ndim = self.train_emb.size(1)
