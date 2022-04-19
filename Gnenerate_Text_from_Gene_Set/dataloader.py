import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils import calculate_jaccard


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class src2tgtData(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return {'source': self.data[item][1],
                'target': self.data[item][0],
                'dist': torch.from_numpy(self.data[item][2]).float()}

    def __len__(self):
        return len(self.data)


class KNNData:
    def __init__(self, opt):
        self.opt = opt
        self.raw_data = pd.DataFrame(load_obj(opt.raw_path))
        self.raw_idx = self.raw_data.index
        self.all_features = [self.raw_data.loc[i]['features'].reshape([-1]) for i in self.raw_idx]
        self.all_text = [self.raw_data.loc[i]['def'] for i in self.raw_idx]
        self.all_prots = [self.raw_data.loc[i]['prot_list'] for i in self.raw_idx]
        self.jaccard_matrix = np.zeros([len(self.all_features), len(self.all_features)])
        print('Calculate jaccard similarity......\n')
        for i in tqdm(range(len(self.all_features))):
            for j in range(i, len(self.all_features)):
                jac_ij = calculate_jaccard(self.all_prots[i], self.all_prots[j])
                self.jaccard_matrix[i, j] = jac_ij
                self.jaccard_matrix[j, i] = jac_ij

    def generate_data(self, random_state=42):
        train_idx, test_idx = train_test_split(np.arange(len(self.raw_idx)), test_size=self.opt.test_size, random_state=random_state)
        exclude_idx = []
        print('Exclude terms with jaccard similarity greater than: {}'.format(self.opt.exclude_jac))
        for i in tqdm(range(len(test_idx))):
            jac_i = np.max(self.jaccard_matrix[test_idx[i], train_idx])
            if jac_i < self.opt.exclude_jac:
                exclude_idx.append(test_idx[i])
        sort_idx = np.argsort(-self.jaccard_matrix, axis=1)
        data = []
        print('Searh {} nearest features......\n'.format(self.opt.K))
        for i in tqdm(range(len(self.all_features))):
            text_i = self.all_text[i]
            text_knn = []
            dist_knn = []
            j = 0
            while True:
                if sort_idx[i, j + 1] in train_idx:
                    text_knn.append(self.all_text[sort_idx[i, j + 1]])
                    dist_knn.append(self.jaccard_matrix[i, sort_idx[i, j + 1]])
                    if len(text_knn) == self.opt.K:
                        break
                j += 1
            data.append((text_i, text_knn, (1 / np.sum(dist_knn))*np.asarray(dist_knn)))
        data = np.array(data, dtype=object)
        return data[train_idx], data[exclude_idx]

