import os
import torch
import collections
import numpy as np
from tqdm import tqdm
from .BioConfig import BioConfig
from .BioLoader import BioLoader
from torch.utils.data import DataLoader
from .BioMetrics import compute_roc, micro_auroc
from .BioModel import BioTranslator, BaseModel, DeepGOPlus
from .BioUtils import sample_edges, get_logger, edge_probability
from .BioUtils import get_few_shot_namespace_terms, save_obj, compute_blast_preds, load_obj


class BioTrainer:

    def __init__(self, files: BioLoader, cfg: BioConfig):
        '''
        This class mainly include the training of BioTranslator Model
        :param files:
        '''
        # pass the dataset details to the config class
        cfg.network_dim = files.network_dim

    def setup_model(self, files: BioLoader, cfg: BioConfig):
        if cfg.method == 'BioTranslator':
            self.model = BioTranslator(cfg)
        elif cfg.method == 'ProTranslator':
            self.model = BaseModel(cfg)
        elif cfg.method == 'TFIDF':
            self.model = BaseModel(cfg)
        elif cfg.method == 'clusDCA':
            self.model = BaseModel(cfg)
        elif cfg.method == 'Word2Vec':
            self.model = BaseModel(cfg)
        elif cfg.method == 'Doc2Vec':
            self.model = BaseModel(cfg)
        elif cfg.method == 'DeepGOPlus':
            self.model = DeepGOPlus(cfg, files.n_classes)
        else:
            print('Warnings: No such method to setup the model!')

    def setup_training(self, files: BioLoader, cfg: BioConfig):
        # train epochs
        self.epoch = cfg.epoch
        # loss function
        self.loss_func = torch.nn.BCELoss()
        # use Adam as optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        # set up the training dataset and validation dataset
        self.train_dataset = files.train_data
        self.val_dataset = files.eval_data

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        logits = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(logits, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def train(self, files: BioLoader, cfg: BioConfig):
        # store the results of pathway prediction
        node_link_auroc = collections.OrderedDict()
        # record the performance of our model
        self.logger = get_logger(cfg.logger_name)

        print('Start training model on :{} ...'.format(cfg.dataset))

        # setup the model architecture according the methods
        self.setup_model(files, cfg)
        # setup dataset, the training loss and optimizer
        self.setup_training(files, cfg)

        train_loader = DataLoader(files.train_data, batch_size=cfg.batch_size, shuffle=True)
        eval_loader = DataLoader(files.eval_data, batch_size=cfg.batch_size, shuffle=True)

        # train the model
        pbar = tqdm(range(self.epoch), desc='Training')
        for epoch_i in pbar:
            train_loss = 0
            train_count = 0
            for batch in train_loader:
                train_loss += self.backward_model(batch['prot_seq'], batch['prot_description'],
                                    batch['prot_network'], files.text_embeddings, batch['label'])
                train_count += len(batch)
                pbar.set_postfix({"epoch": epoch_i,
                                  "train loss": train_loss / train_count})

        # evaluate the model
        print('Evaluate Our Model on {}'.format(cfg.pathway_dataset))
        uniprots, preds, label = [], [], []
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc='Pathway Node Classification')
            for batch in pbar:
                pred_batch = self.model.data_encoder(batch['prot_seq'], batch['prot_description'],
                                        batch['prot_network'])
                uniprots += batch['proteins']
                self.loss_func.zero_grad()
                preds.append(pred_batch.cpu().numpy())
                label.append(batch['label'].cpu().float().numpy())

        data_encodings = np.concatenate(preds, axis=0)
        pathway_labels = np.concatenate(label, axis=0)
        pathway_encodings = files.pathway_embeddings.cpu().numpy()
        go_embeddings = files.text_embeddings.cpu().numpy()
        nearest_k = 5
        sims = np.dot(pathway_encodings, np.transpose(go_embeddings))
        for i in range(np.size(pathway_encodings, 0)):
            sim_i = sims[i, :]
            sim_idx = np.argsort(-sim_i)[0: nearest_k]
            nearest_go_emb = go_embeddings[sim_idx, :]
            nearest_go_emb = np.vstack((go_embeddings[i, :], nearest_go_emb))
            emb_i = np.mean(nearest_go_emb, axis=0)
            pathway_encodings[i, :] = emb_i / np.sqrt(np.sum(np.power(emb_i, 2)))
        logits = np.dot(data_encodings, np.transpose(pathway_encodings))
        logits = 1 / (1 + np.exp(-logits))

        # evaluate the performance of BioTranslator on node classification
        auroc = micro_auroc(pathway_labels, logits)
        for i in range(len(auroc)):
            node_link_auroc[files.eval_i2terms[i]] = [auroc[i]]

        self.logger.info('Pathway: {} Node Classification AUROC: {}'.format(cfg.pathway_dataset, np.mean(auroc)))

        # evaluate the performance of BioTranslator on edge prediction
        pathway_dir = cfg.data_repo + cfg.pathway_dataset + '/'
        if 'pathway_graph.pkl' in os.listdir(pathway_dir):
            pathway_graph = load_obj(pathway_dir + 'pathway_graph.pkl')
            uniprot2index = collections.OrderedDict()
            id = 0
            for uniprot_id in uniprots:
                uniprot2index[uniprot_id] = id
                id += 1

            graph_auroc = []
            pbar = tqdm(pathway_graph.keys(), desc='Pathway Edge Prediction')
            for p_id in pbar:
                edges_id, edges = [], pathway_graph[p_id]

                # convert edges from uniprot to index
                for e in range(len(edges)):
                    if edges[e][0] not in uniprot2index.keys() or edges[e][1] not in uniprot2index.keys():
                        continue
                    edges_id.append([uniprot2index[edges[e][0]], uniprot2index[edges[e][1]]])
                if len(edges_id) == 0:
                    continue
                false_edges = sample_edges(0, len(uniprots), 10 * len(edges_id), edges_id)
                true_edges = np.asarray(edges_id)

                # predict edges
                encodings_i, encodings_j = data_encodings[true_edges[:, 0], :], data_encodings[true_edges[:, 1], :]
                text_encodings_p = pathway_encodings[files.eval_terms2i[p_id]] / np.sqrt(
                    np.sum(np.square(pathway_encodings[files.eval_terms2i[p_id]].squeeze())))
                text_encodings_p = text_encodings_p.reshape([1, -1])
                true_score = edge_probability(encodings_i, encodings_j, text_encodings_p)
                encodings_i = data_encodings[false_edges[:, 0], :]
                encodings_j = data_encodings[false_edges[:, 1], :]
                false_score = edge_probability(encodings_i, encodings_j, text_encodings_p)

                # compute auroc
                edge_prediction, edge_label = [], []
                edge_prediction += list(true_score)
                edge_prediction += list(false_score)
                edge_label += list(np.ones(len(true_score)))
                edge_label += list(np.zeros(len(false_score)))
                edge_auroc = compute_roc(np.asarray(edge_label), np.asarray(edge_prediction))
                graph_auroc.append(edge_auroc)
                node_link_auroc[p_id].append(edge_auroc)
            self.logger.info('Pathway: {} Edge Prediction AUROC: {}'.format(cfg.pathway_dataset, np.mean(graph_auroc)))

        # save the results
        save_obj(node_link_auroc, cfg.results_name)
        # save the model
        torch.save(self.model, cfg.save_model_path.format(cfg.method, cfg.pathway_dataset))








