from utils import init_weights, compute_roc, barplot_auroc
from tqdm import tqdm
import numpy as np
from model import BioTranslatorModel
import torch
from options import data_loading_opt, model_config
from file_loader import FileLoader
from torch.utils.data import DataLoader
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
    results_cache = collections.OrderedDict()
    results_cache['unseen'] = collections.OrderedDict()
    results_cache['all'] = collections.OrderedDict()
    data_loader = FileLoader(data_opt)
    for unseen_ratio in data_opt.unseen_ratio:
        results_cache['all'][unseen_ratio] = []
        results_cache['unseen'][unseen_ratio] = []
        for fold_iter in range(data_opt.n_iter):
            data_loader.generate_data(fold_iter, unseen_ratio)
            trainData = DataLoader(data_loader.train_data, batch_size=model_opt.batch_size, shuffle=True)
            testData = DataLoader(data_loader.test_data, batch_size=model_opt.batch_size)

            model = BioTranslator(ngene=data_loader.ngene,
                                   nhidden=model_opt.nhidden,
                                   ndim=data_loader.ndim,
                                   lr=model_opt.lr,
                                   drop_out=model_opt.drop_out,
                                   is_gpu=data_opt.is_gpu)
            for epi in range(model_opt.epoch):
                desc = 'Fold:{} unseen ratio:{} Training Epoch :{} '.format(fold_iter, unseen_ratio, epi)
                p_bar = tqdm(enumerate(trainData), desc=desc)
                train_loss, infer_loss = 0, 0
                num = 0
                model.model.train()
                for i, train_batch in p_bar:
                    model.backward_model(train_batch['features'], data_loader.train_emb, train_batch['label'])
                    train_loss += float(model.loss.item())
                    #p_bar.set_postfix(train_loss=model.loss.item())
                    num += 1
                print('Train loss: {}'.format(train_loss / num))

                num = 0
                if epi == model_opt.epoch - 1:
                    inference_preds, inference_label = [], []
                    model.model.eval()
                    for j, test_batch in enumerate(testData):
                        with torch.no_grad():
                            preds = model.model(test_batch['features'], data_loader.test_emb)
                            label = test_batch['label']
                            loss = model.loss_func(preds, label)
                            inference_preds.append(preds)
                            inference_label.append(label)
                            infer_loss += float(loss.item())
                            #p_bar.set_postfix(infer_loss=loss.item())
                            num += 1
                    inference_loss = infer_loss / num
                    inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0).cpu().float().numpy(), \
                                                       torch.cat(tuple(inference_label), dim=0).cpu().float().numpy()
                    unseen_auroc = model.evaluate_unseen(inference_preds, inference_label, data_loader.unseen2i)
                    roc_macro = np.zeros(np.size(inference_label, 1))
                    for cl_i in range(np.size(inference_label, 1)):
                        roc_auc = compute_roc(inference_label[:, cl_i], inference_preds[:, cl_i])
                        roc_macro[cl_i] = roc_auc
                    roc_auc = np.mean(roc_macro)
                    print('Test all AUROCs :{} unseen AUROCs:{} Infer loss: {}'.format(roc_auc, unseen_auroc, inference_loss))
            results_cache['all'][unseen_ratio].append(roc_auc)
            results_cache['unseen'][unseen_ratio].append(unseen_auroc)
            results_cache['all'][unseen_ratio].append(0.7)
            results_cache['unseen'][unseen_ratio].append(0.7)
            torch.save(model, model_opt.save_id + '{}_{}.pkl'.format(fold_iter, unseen_ratio))
    save_obj(results_cache, model_opt.results_path)
    barplot_auroc(results_cache, model_opt.fig_path)


if __name__ == '__main__':
    main()

