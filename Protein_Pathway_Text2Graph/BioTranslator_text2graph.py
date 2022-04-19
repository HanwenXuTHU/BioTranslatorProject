import collections
import numpy as np
import pdb
import pickle
import torch
from model import BioTranslatorModel, Text2GraphModel
from options import model_config, data_loading, Text2Graph_config
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from utils import init_weights, EarlyStopping, compute_roc, vec2classes


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class BioTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = BioTranslatorModel.BioTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            self.model = self.model.cuda()
            init_weights(self.model, init_type='xavier')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, emb_tensor, label):
        preds = self.model(input_seq, emb_tensor)
        pdb.set_trace()
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class Text2Graph:

    def __init__(self, Text2Graph_config):
        self.loss_func = CrossEntropyLoss()
        self.model = Text2GraphModel.LinkPrediction(input_nc=Text2Graph_config.input_nc,
                                                    x_dim=Text2Graph_config.x_dim,
                                                    graph_emb_dim=Text2Graph_config.graph_emb_dim,
                                                    num_layers=Text2Graph_config.num_layers,
                                                    GAT_heads=Text2Graph_config.GAT_heads,
                                                    with_graph=Text2Graph_config.with_graph,
                                                    with_text=Text2Graph_config.with_text,
                                                    seqL=Text2Graph_config.seqL)
        if len(Text2Graph_config.gpu_ids) > 0:
            self.model = self.model.cuda()
            #init_weights(self.model, init_type='xavier')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Text2Graph_config.lr)

    def backward_model(self, data):
        preds, labels = self.model(data)
        self.loss = self.loss_func(preds, labels)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def extract_sub_df(df, col_str, value):
    extract_list = []
    for i in df.index:
        if value in df.loc[i][col_str]: extract_list.append(i)
    return df.loc[extract_list]


def extract_gene_features(df, descriptions, vector):
    gene2feature = collections.OrderedDict()
    for i in df.index:
        if df.loc[i]['proteins'] in descriptions.keys() and df.loc[i]['proteins'] in vector.keys():
            gene2feature[df.loc[i]['GN']] = (one_hot(df.loc[i]['sequences']), \
                                        descriptions[df.loc[i]['proteins']], \
                                        vector[df.loc[i]['proteins']])
    return gene2feature


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


class ComplementGraph(object):

    def __init__(self, edge_index, node_list):
        super(ComplementGraph, self).__init__()
        self.node_num = len(node_list)
        self.edge_set = {}
        self.node_list = node_list
        for f,t in edge_index:
            self.update(f, t)

    def update(self, f, t):
        if not (f in self.edge_set.keys()):
            self.edge_set[f] = set()
        self.edge_set[f].add(t)

    def exist(self, f, t):
        if not (f in self.edge_set.keys()):
            return False
        return t in self.edge_set[f]

    def generate(self, n):
        pairs = []
        for i in range(n):
            while(True):
                f = np.random.choice(a=self.node_list, size=1, replace=False)[0]
                t = np.random.choice(a=self.node_list, size=1, replace=False)[0]
                if f == t:
                    continue
                if(not self.exist(f, t)):
                    pairs.append([f, t])
                    break
        return pairs


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class graphData(Dataset):

    def __init__(self, pathway_list,
                 pathway_dataset, pathway_vector, pathway_description,
                 pathway_network, term_emb,
                 seen_ratio=0.6,
                 model_path='',
                 gpu_ids='0'):
        self.graph_list = []
        self.gpu_ids = gpu_ids
        self.gene2features = extract_gene_features(pathway_dataset, pathway_description, pathway_vector)
        seed_i = 0
        for path_i in pathway_list:
            subset_genes = []
            path_i_edges = []
            node_list = []
            edge_attr = []
            for id1 in pathway_network[path_i].keys():
                if id1 not in self.gene2features.keys():
                    continue
                else:
                    subset_genes.append(id1)
                for id2 in pathway_network[path_i][id1]:
                    if id2 not in self.gene2features.keys():
                        continue
                    else:
                        path_i_edges.append([id1, id2])
                        subset_genes.append(id2)
            if len(path_i_edges) <= 50: continue
            seen_edges_mask = np.arange(len(path_i_edges)) <= int(seen_ratio*len(path_i_edges))
            np.random.seed(seed_i)
            seed_i += 1
            np.random.shuffle(seen_edges_mask)
            seen_edges = np.asarray(path_i_edges)[seen_edges_mask]
            true_edges = np.asarray(path_i_edges)[seen_edges_mask==False]
            cGraph = ComplementGraph(path_i_edges, list(subset_genes))
            false_edges = cGraph.generate(int((1 - seen_ratio) * len(path_i_edges)))
            unseen_edges = np.vstack((true_edges, false_edges))
            seen_edges_idx, unseen_edges_idx, seen_labels, unseen_labels = [], [], [], []
            subset_genes = list(set(subset_genes))
            gene2i = collections.OrderedDict()
            idx = 0
            for gene in subset_genes:
                gene2i[gene] = idx
                idx += 1
            for i in range(len(seen_edges)):
                seen_labels.append(1)
                edge_attr.append(np.ones(1))
                seen_edges_idx.append([gene2i[seen_edges[i, 0]], gene2i[seen_edges[i, 1]]])
            for i in range(len(unseen_edges)):
                unseen_edges_idx.append([gene2i[unseen_edges[i, 0]], gene2i[unseen_edges[i, 1]]])
                if i < len(true_edges):
                    unseen_labels.append(1)
                    edge_attr.append(np.ones(1))
                else:
                    unseen_labels.append(0)
                    edge_attr.append(np.zeros(1))
            for gene in gene2i.keys():
                node_list.append(self.gene2features[gene])
            self.graph_list.append({'nodes': node_list,
                                    'seen_edges': seen_edges_idx,
                                    'unseen_edges': unseen_edges_idx,
                                    'seen_labels': seen_labels,
                                    'unseen_labels': unseen_labels,
                                    'edge_attr': edge_attr,
                                    'text_emb': term_emb[path_i]})
            self.model = torch.load(model_path)

    def update_feature_emb(self):
        self.data_list = []
        for i in tqdm(range(len(self.graph_list))):
            graph = self.graph_list[i]
            nodes = graph['nodes']
            x = []
            for j in range(len(nodes)):
                seq, description, vector = nodes[j]
                seq = torch.from_numpy(seq).float().unsqueeze(dim=0)
                description = torch.from_numpy(description).float()
                if len(self.gpu_ids) > 0:
                    seq, description = seq.cuda(), description.cuda()
                with torch.no_grad():
                    node_emb = self.model.model.get_emb(x=seq, x_description=description)
                emb = torch.squeeze(node_emb).cpu().float().numpy()
                emb = emb.reshape([-1])
                emb = (emb - np.mean(emb)) / np.std(emb)
                x.append(emb.reshape([1, -1]))
            x = np.vstack(x)
            y = self.graph_list[i]['text_emb'].reshape([-1])
            edge_index = self.graph_list[i]['seen_edges'] + self.graph_list[i]['unseen_edges']
            edge_index = np.asarray(edge_index).transpose()
            edge_labels = self.graph_list[i]['seen_labels'] + self.graph_list[i]['unseen_labels']
            train_edge_mask = np.arange(len(edge_labels)) < len(self.graph_list[i]['seen_labels'])
            train_edge_mask = torch.tensor(train_edge_mask, dtype=torch.bool)
            edge_labels = torch.tensor(edge_labels, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(self.graph_list[i]['edge_attr'], dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)
            y = y / np.sqrt(np.sum(np.square(y)))
            y = torch.tensor([y], dtype=torch.float)
            if len(self.gpu_ids) > 0:
                x, y, edge_index = x.cuda(), y.cuda(), edge_index.cuda()
                edge_labels, train_edge_mask = edge_labels.cuda(), train_edge_mask.cuda()
            data_i = Data(x=x, y=y, edge_index=edge_index,
                          edge_attr=edge_attr,
                          edge_label=edge_labels,
                          train_edge_mask=train_edge_mask)
            self.data_list.append(data_i)


def main():
    dataset = 'KEGG'
    nFold = 3
    batch_size = 32
    data_opt = data_loading()
    root_dir = data_opt.pathway_dir
    gpu_ids = data_opt.gpu_ids
    model_opt = model_config()
    t2g_config = Text2Graph_config()
    data_dir = dataset
    print('Loading dataset {}'.format(dataset))
    pathway_dataset = load_obj('{}/{}/dataset.pkl'.format(root_dir, data_dir))
    pathway_term_dict = load_obj('{}/{}/terms.pkl'.format(root_dir, data_dir))
    pathway_prot_vector = load_obj('{}/{}/prot_vector.pkl'.format(root_dir, data_dir))
    pathway_prot_description = load_obj('{}/{}/prot_description.pkl'.format(root_dir, data_dir))
    term_emb = load_obj('{}/{}/text_emb.pkl'.format(root_dir, data_dir))
    pathway_network = load_obj('{}/{}/networks.pkl'.format(root_dir, data_dir))
    pathway_list = []
    for pathway_i in pathway_network.keys():
        if len(pathway_network[pathway_i].keys()) > 0 and pathway_i in list(pathway_term_dict['terms']):
            pathway_list.append(pathway_i)

    graph_data = graphData(pathway_list, pathway_dataset,
                                   pathway_prot_vector, pathway_prot_description,
                                   pathway_network,
                                   term_emb,
                                   gpu_ids=gpu_ids,
                                   seen_ratio=t2g_config.seen_ratio,
                                   model_path=model_opt.save_path.format(data_opt.dataset))
    graph_data.update_feature_emb()
    results = collections.OrderedDict()
    for test_ratio in t2g_config.test_ratio:
        results[test_ratio] = []
        for fold_i in range(nFold):
            train_idx, test_idx = train_test_split(np.arange(len(graph_data.data_list)), test_size=test_ratio, random_state=fold_i)
            train_data, val_data = [], []
            for i in range(len(train_idx)): train_data.append(graph_data.data_list[i])
            for i in range(len(test_idx)): val_data.append(graph_data.data_list[i])
            train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
            val_data_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
            model = Text2Graph(t2g_config)
            model.model.train()
            for i in range(t2g_config.epoch):
                loss, num = 0, 0
                val_loss, val_num = 0, 0
                inference_preds, inference_label = [], []
                for train_batch in train_data_loader:
                    model.backward_model(train_batch)
                    loss += model.loss.item()
                    num += len(train_batch)
                with torch.no_grad():
                    for val_batch in val_data_loader:
                        preds, labels = model.model(val_batch)
                        val_loss += model.loss_func(preds, labels).item()
                        val_num += len(val_batch)
                        inference_preds.append(preds)
                        inference_label.append(labels)
                inference_preds, inference_label = torch.cat(tuple(inference_preds), dim=0), torch.cat(tuple(inference_label), dim=0)
                inference_preds, inference_label = inference_preds.cpu().float().numpy(), inference_label.cpu().float().numpy()
                roc_auc = compute_roc(inference_label[:], inference_preds[:, 1])
                print('epoch:{} fold:{} test ratio: {}, Train loss: {}, val loss: {}, val auroc: {}\n'
                      .format(i, fold_i, test_ratio, loss / num, val_loss / val_num, roc_auc))
            results[test_ratio].append(roc_auc)
        save_obj(results, t2g_config.save_auroc)


if __name__ == '__main__':
    main()