import torch
import torch.nn as nn
from model import BioTranslatorModel
from options import model_config, data_loading, pathway_config
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, auroc_bar_percentage_plot
import utils
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from torch import squeeze
import pickle
import collections


class BioTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = BioTranslatorModel.BioTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                vector_dim=model_config.N_vector_dim,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            init_weights(self.model, init_type='xavier')
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        preds = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def one_hot(seq, start=0, max_len=2000):
    AALETTER = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot


class proteinData(Dataset):
    def __init__(self, data_df, terms, prot_vector, prot_description, gpu_ids='0'):
        self.pSeq = []
        self.label = []
        self.vector = []
        self.description = []
        self.gpu_ids = gpu_ids
        sequences = list(data_df['sequences'])
        prot_ids = list(data_df['proteins'])
        annots = list(data_df['annotations'])
        for i in range(len(annots)):
            annots[i] = list(annots[i])
        for i in range(data_df.shape[0]):
            seqT, annT, protT = sequences[i], annots[i], prot_ids[i]
            labelT = np.zeros([len(terms), 1])
            for j in range(len(annT)):
                if annT[j] in terms.keys():
                    labelT[terms[annT[j]]] = 1
            self.pSeq.append(one_hot(seqT))
            self.label.append(labelT)
            self.vector.append(prot_vector[protT])
            self.description.append(prot_description[protT])

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        description = torch.from_numpy(self.description[item])
        vector = torch.from_numpy(self.vector[item])
        if len(self.gpu_ids) > 0:
            return {'seq': squeeze(in_seq).float().cuda(),
                    'description':squeeze(description).float().cuda(),
                    'vector':squeeze(vector).float().cuda(),
                    'label': squeeze(label).float().cuda()}
        else:
            return {'seq': squeeze(in_seq).float(),
                    'description': squeeze(description).float(),
                    'vector': squeeze(vector).float(),
                    'label': squeeze(label).float()}

    def __len__(self):
        return len(self.pSeq)


def emb2tensor(brief_embeddings, full_embeddings, terms, text_mode='brief'):
    ann_id = list(terms.keys())
    print('Text mode is {}'.format(text_mode))
    if text_mode == 'brief':
        embedding_array = np.zeros((len(ann_id), np.size(brief_embeddings[ann_id[0]], 1)))
    elif text_mode == 'full':
        embedding_array = np.zeros((len(ann_id), np.size(full_embeddings[ann_id[0]], 1)))
    elif text_mode == 'both':
        embedding_array = np.zeros((len(ann_id), np.size(full_embeddings[ann_id[0]], 1) + np.size(brief_embeddings[ann_id[0]], 1)))
    for t_id in ann_id:
        if text_mode == 'brief':
            t_brief = brief_embeddings[t_id].reshape([1, -1])
            t_brief = t_brief / np.sqrt(np.sum(np.power(t_brief, 2), axis=1))
            embedding_array[terms[t_id], :] = t_brief
        elif text_mode == 'full':
            t_full = full_embeddings[t_id].reshape([1, -1])
            t_full = t_full / np.sqrt(np.sum(np.power(t_full, 2), axis=1))
            embedding_array[terms[t_id], :] = t_full
        elif text_mode == 'both':
            t_full = full_embeddings[t_id].reshape([1, -1])
            t_brief = brief_embeddings[t_id].reshape([1, -1])
            t = np.hstack((t_brief, t_full))
            t = t / np.sqrt(np.sum(np.power(t, 2), axis=1))
            embedding_array[terms[t_id], :] = t
    rank_e = np.linalg.matrix_rank(embedding_array)
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    data_opt = data_loading()
    model_opt = model_config()
    pathway_opt = pathway_config()
    torch.cuda.set_device('cuda:' + data_opt.gpu_ids)
    root_dir = pathway_opt.data_dir
    dataset = pathway_opt.dataset
    model_predict = torch.load(model_opt.save_path.format(data_opt.dataset))
    dataset_path = '{}/{}/dataset.pkl'.format(root_dir, dataset)
    terms_path = '{}/{}/terms.pkl'.format(root_dir, dataset)
    emb_path = '{}/{}/textual_description_embeddings.pkl'.format(root_dir, dataset)
    infer_data = pd.read_pickle(dataset_path)
    terms = pd.read_pickle(terms_path)
    embeddings = pd.read_pickle(emb_path)
    prot_vector = load_obj('{}/{}/prot_vector.pkl'.format(root_dir, dataset))
    prot_description = load_obj('{}/{}/prot_description.pkl'.format(root_dir, dataset))
    term_list = list(terms['terms'])
    infer_terms = {term_list[i]: i for i in range(len(term_list))}
    inference_dataset = proteinData(infer_data, infer_terms, prot_vector=prot_vector, prot_description=prot_description, gpu_ids=data_opt.gpu_ids)
    inference_dataset = DataLoader(inference_dataset, batch_size=model_opt.batch_size, shuffle=True)

    emb_pathway_tensor = emb2tensor(brief_embeddings=None, full_embeddings=embeddings, terms=infer_terms, text_mode='full')
    if len(data_opt.gpu_ids) > 0: emb_pathway_tensor = emb_pathway_tensor.float().cuda()
    print('inferring iters')
    inference_preds, inference_label = [], []
    for j, inference_D in tqdm(enumerate(inference_dataset)):
        with torch.no_grad():
            preds = model_predict.model(inference_D['seq'], inference_D['description'], inference_D['vector'], emb_pathway_tensor)
            loss = model_predict.loss_func(preds, inference_D['label'])
            inference_preds.append(preds)
            inference_label.append(inference_D['label'])
    inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
    inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()

    percentage = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    auroc_percentage = collections.OrderedDict()
    for p in percentage: auroc_percentage[p] = 0
    infer_terms_auroc = collections.OrderedDict()
    for t_id in infer_terms.keys():
        j = infer_terms[t_id]
        preds_id = inference_preds[:, j].reshape([-1, 1])
        label_id = inference_label[:, j].reshape([-1, 1])
        roc_auc_j = compute_roc(label_id, preds_id)
        for T in auroc_percentage.keys():
            if T <= roc_auc_j:
                auroc_percentage[T] += 100.0 / np.size(inference_preds, 1)
        infer_terms_auroc[t_id] = roc_auc_j
    for T in auroc_percentage.keys():
        print('roc auc:{} percentage:{}%'.format(T, auroc_percentage[T]))
    save_obj(auroc_percentage, pathway_opt.auroc_geq_threshold_barplot_save_path)
    auroc_bar_percentage_plot(auroc_percentage, pathway_opt.auroc_geq_threshold_barplot_save_path)



if __name__ == '__main__':
    main()