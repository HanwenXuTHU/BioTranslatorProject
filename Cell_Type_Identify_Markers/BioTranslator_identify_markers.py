from utils import init_weights, compute_roc, corr2_coeff, read_type2genes
import numpy as np
from model import BioTranslatorModel
import scipy
import torch
from options import data_loading_opt, model_config
from file_loader import FileLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
import collections
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class BioTranslator:

    def __init__(self,
                 ngene=2000,
                 nhidden=[1000],
                 ndim=100,
                 lr=0.003,
                 drop_out=0.1,
                 l2=0,
                 is_gpu=True):
        self.loss_func = torch.nn.BCELoss()
        self.model = BioTranslatorModel.BioTranslatorModel(ngene=ngene, nhidden=nhidden, ndim=ndim, drop_out=drop_out)
        if is_gpu:
            self.model = self.model.cuda()
            init_weights(self.model, init_type='xavier')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

    def backward_model(self, features, emb_tensor, label):
        preds = self.model(features, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def evaluate_unseen(self, inference_preds, inference_label, unseen2i):
        roc_macro = np.zeros(len(unseen2i.values()))
        unseen_id = list(unseen2i.values())
        cl_i = 0
        for i in unseen_id:
            unseen_preds, unseen_labels = inference_preds[:, i], inference_label[:, i]
            auroc = compute_roc(unseen_labels, unseen_preds)
            roc_macro[cl_i] = auroc
            cl_i += 1
        return np.mean(roc_macro)


def main():
    data_opt = data_loading_opt()
    model_opt = model_config()
    marker_preds = collections.OrderedDict()
    data_loader = FileLoader(data_opt)
    for unseen_ratio in data_opt.unseen_ratio:
        for fold_iter in [0]:
            data_loader.generate_data(fold_iter, unseen_ratio)
            trainData = DataLoader(data_loader.train_data, batch_size=model_opt.batch_size, shuffle=False)
            testData = DataLoader(data_loader.test_data, batch_size=model_opt.batch_size, shuffle=False)
            if data_loader.opt.memory_saving_mode:
                train_feature = np.asarray(data_loader.train_data.expression_matrix.todense()).squeeze()
                test_feature = np.asarray(data_loader.test_data.expression_matrix.todense()).squeeze()
            else:
                train_feature = data_loader.train_data.expression_matrix
                test_feature = data_loader.test_data.expression_matrix
            train_feature = scipy.sparse.vstack((train_feature, test_feature))
            model = torch.load(model_opt.save_id + '{}_{}.pkl'
                               .format(fold_iter, unseen_ratio))
            inference_preds, inference_label = [], []
            model.model.eval()
            for j, train_batch in enumerate(trainData):
                with torch.no_grad():
                    preds = model.model(train_batch['features'], data_loader.test_emb)
                    inference_preds.append(preds)
            for j, test_batch in enumerate(testData):
                with torch.no_grad():
                    preds = model.model(test_batch['features'], data_loader.test_emb)
                    inference_preds.append(preds)
            inference_preds = torch.cat(tuple(inference_preds), dim=0).cpu().float().numpy()
            #cor = corr2_coeff(inference_preds[:, :].T, train_feature.tocsr().T)
            cor_list = []
            train_feature = train_feature.tocsr()
            batch_n = 3000
            for i in tqdm(range(100)):
                st = i*batch_n
                ed = min((i+1)*batch_n, train_feature.shape[1])
                feature_i = train_feature[:, st : ed].toarray().reshape([-1, ed - st]).T
                cor_i = corr2_coeff(inference_preds[:, :].T, feature_i).reshape([-1, ed-st])
                cor_list.append(cor_i)
                if ed == train_feature.shape[1]:
                    break
            cor = np.hstack(tuple(cor_list))
            marker_preds['genes'] = data_loader.train_genes
            marker_preds['i2test'] = data_loader.i2test
            marker_preds['i2unseen'] = data_loader.i2unseen
            for i in range(np.size(cor, 0)):
                marker_preds[data_loader.i2test[i]] = cor[i, :]
    save_obj(marker_preds, model_opt.save_marker_path)


if __name__ == '__main__':
    main()

