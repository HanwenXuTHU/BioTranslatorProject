import os
import sys
import torch
import logging
import time
import pickle
import collections
import numpy as np
import pandas as pd
from torch import nn
import networkx as nx
from tqdm import tqdm
from torch import squeeze
from torch.nn import init
from collections import deque
from gensim import corpora, models
from scipy.sparse.linalg import svds
from nltk.tokenize import word_tokenize
from torch.utils.data import ConcatDataset
from gensim.models.word2vec import Word2Vec
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load(filename, with_rels=True):
    ont = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['alt_ids'] = list()
                obj['def'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'alt_id':
                    obj['alt_ids'].append(l[1])
                elif l[0] == 'namespace':
                    obj['namespace'] = l[1]
                elif l[0] == 'def':
                    obj['def'].append(l[1].split('"')[1])
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif with_rels and l[0] == 'relationship':
                    it = l[1].split()
                    # add all types of relationships
                    obj['is_a'].append(it[1])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
    for term_id in list(ont.keys()):
        for t_id in ont[term_id]['alt_ids']:
            ont[t_id] = ont[term_id]
        if ont[term_id]['is_obsolete']:
            del ont[term_id]
    for term_id, val in ont.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in ont:
                if 'children' not in ont[p_id]:
                    ont[p_id]['children'] = set()
                ont[p_id]['children'].add(term_id)
    return ont


def get_anchestors(go, term_id):
    if term_id not in go:
        return []
    term_set = []
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.append(t_id)
            for parent_id in go[t_id]['is_a']:
                if parent_id in go:
                    q.append(parent_id)
    return term_set


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_logger(logfile):
    '''
    This function are copied from https://blog.csdn.net/a232884c/article/details/117453011
    :param logfile:
    :return:
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def RandomWalkRestart(A, rst_prob, delta = 1e-4, reset=None, max_iter=50,use_torch=False,return_torch=False):
    if use_torch:
        device = torch.device("cuda:0")
    nnode = A.shape[0]
    if reset is None:
        reset = np.eye(nnode)
    nsample, nnode = reset.shape
    #print nsample,nnode
    P = renorm(A)
    P = P.T
    norm_reset = renorm(reset.T)
    norm_reset = norm_reset.T
    if use_torch:
        norm_reset = torch.from_numpy(norm_reset).float().to(device)
        P = torch.from_numpy(P).float().to(device)
    Q = norm_reset

    for i in range(1,max_iter):
        if use_torch:
            Q_new = rst_prob*norm_reset + (1-rst_prob) * torch.mm(Q, P)#.as_numpy_array()
            delta = torch.norm(Q-Q_new, 2)
        else:
            Q_new = rst_prob*norm_reset + (1-rst_prob) * np.dot(Q, P)#.as_numpy_array()
            delta = np.linalg.norm(Q-Q_new, 'fro')
        Q = Q_new
        #print (i,Q)
        sys.stdout.flush()
        if delta < 1e-4:
            break
    if use_torch and not return_torch:
        Q = Q.cpu().numpy()
    return Q


def renorm(X):
    Y = X.copy()
    Y = Y.astype(float)
    ngene,nsample = Y.shape
    s = np.sum(Y, axis=0)
    #print s.shape()
    for i in range(nsample):
        if s[i]==0:
            s[i] = 1
            if i < ngene:
                Y[i,i] = 1
            else:
                for j in range(ngene):
                    Y[j,i] = 1. / ngene
        Y[:,i] = Y[:,i]/s[i]
    return Y


def DCA_vector(Q, dim):
    nnode = Q.shape[0]
    alpha = 1. / (nnode **2)
    Q = np.log(Q + alpha) - np.log(alpha);

    #Q = Q * Q';
    [U, S, V] = svds(Q, dim);
    S = np.diag(S)
    X = np.dot(U, np.sqrt(S))
    Y = np.dot(np.sqrt(S), V)
    Y = np.transpose(Y)
    return X,U,S,V,Y


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def one_hot(seq, start=0, max_len=2000):
    '''
    One-Hot encodings of protein sequences,
    this function was copied from DeepGOPlus paper
    :param seq:
    :param start:
    :param max_len:
    :return:
    '''
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
    '''
    The dataset of protein
    '''
    def __init__(self, data_df, terms, prot_vector, prot_description, gpu_ids='0'):
        self.p_ids = []
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
            self.p_ids.append(protT)
            self.pSeq.append(one_hot(seqT))
            self.label.append(labelT)
            self.vector.append(prot_vector[protT])
            self.description.append((prot_description[protT]))

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        description = torch.from_numpy(self.description[item])
        vector = torch.from_numpy(self.vector[item])
        if len(self.gpu_ids) > 0:
            return {'proteins': self.p_ids[item],
                    'prot_seq': squeeze(in_seq).float().cuda(),
                    'prot_description': squeeze(description).float().cuda(),
                    'prot_network': squeeze(vector).float().cuda(),
                    'label': squeeze(label).float().cuda()}
        else:
            return {'proteins': self.p_ids[item],
                    'prot_seq': squeeze(in_seq).float(),
                    'prot_description': squeeze(description).float(),
                    'prot_network': squeeze(vector).float(),
                    'label': squeeze(label).float()}

    def __len__(self):
        return len(self.pSeq)


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way, bert_name):
        super(NeuralNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output


def emb2tensor(text_embeddings, terms):
    '''
    This function re-sort the textual description embeddings
    according to the mapping between terms and index
    :param text_embeddings:
    :param terms:
    :return:
    '''
    ann_id = list(terms.keys())
    embedding_array = np.zeros((len(ann_id), np.size(text_embeddings[ann_id[0]], 1)))
    for t_id in ann_id:
        t_def = text_embeddings[t_id].reshape([1, -1])
        t_def = t_def / np.sqrt(np.sum(np.power(t_def, 2), axis=1))
        embedding_array[terms[t_id], :] = t_def
    rank_e = np.linalg.matrix_rank(embedding_array)
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array


def gen_go_emb(cfg):
    if cfg.method == 'BioTranslator':
        get_BioTranslator_emb(cfg)
    elif cfg.method == 'ProTranslator':
        get_ProTranslator_emb(cfg)
    elif cfg.method == 'TFIDF':
        get_TFIDF_emb(cfg)
    elif cfg.method == 'clusDCA':
        get_clusDCA_emb(cfg)
    elif cfg.method == 'Word2Vec':
        get_Word2Vec_emb(cfg)
    elif cfg.method == 'Doc2Vec':
        get_Doc2Vec_emb(cfg)
    else:
        print('Warnings: No such method to embed terms!')


def get_BioTranslator_emb(cfg):
    '''
    This function uses the BioTranslator Text Encoder to embed the Gene Ontology terms
    :param cfg:
    :return:
    '''
    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())
    texts = []
    for i in tqdm(range(len(go_classes))):
        with torch.no_grad():
            texts.append(go_data[go_classes[i]]['name'] + '. ' + go_data[go_classes[i]]['def'][0])

    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = NeuralNetwork('None', 'cls', bert_name)
    model.load_state_dict(torch.load(cfg.encoder_path))
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(go_classes))):
            text = texts[i]
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            if len(cfg.gpu_ids) > 0:
                inputs = inputs.to('cuda')
            sents_len = min(inputs['input_ids'].size(1), 512)
            input_ids = inputs['input_ids'][0, 0: sents_len].view(len(inputs['input_ids']), -1).to('cuda')
            attention_mask = inputs['attention_mask'][0, 0: sents_len].view(len(inputs['attention_mask']), -1).to(
                'cuda')
            token_type_ids = inputs['token_type_ids'][0, 0: sents_len].view(len(inputs['token_type_ids']), -1).to(
                'cuda')

            pred = model(input_ids, attention_mask, token_type_ids)
            go_embeddings[go_classes[i]] = np.asarray(pred.cpu()).reshape([-1, 768])
        save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def get_ProTranslator_emb(cfg):
    '''
    This function uses the PubMedBert as encoder to embed the Gene Ontology terms
    :param cfg:
    :return:
    '''
    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())
    texts = []
    for i in tqdm(range(len(go_classes))):
        with torch.no_grad():
            texts.append(go_data[go_classes[i]]['name'] + '. ' + go_data[go_classes[i]]['def'][0])

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name)
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(go_classes))):
            text = texts[i]
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            if len(cfg.gpu_ids) > 0:
                inputs = inputs.to('cuda')
            outputs = model(inputs.data['input_ids'][:, 0: min(512, inputs.data['input_ids'].shape[1])])
            second_to_last = torch.mean(outputs.last_hidden_state, 1)
            output_hid = np.asarray(second_to_last.cpu()).reshape([-1, 768])
            go_embeddings[go_classes[i]] = output_hid
        save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def get_TFIDF_emb(cfg):
    '''
    Get the textual description embeddings of the Gene Ontology terms
    :param cfg:
    :return:
    '''
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())
    texts = []
    for i in tqdm(range(len(go_classes))):
        with torch.no_grad():
            text = go_data[go_classes[i]]['name'] + '. ' + go_data[go_classes[i]]['def'][0]
            texts.append(word_tokenize(text))

    dictionary = corpora.Dictionary(texts)
    new_corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(new_corpus)

    tfidf_vec = []
    for i in range(len(texts)):
        string_tfidf = tfidf[dictionary.doc2bow(texts[i])]
        tfidf_vec.append(string_tfidf)
    lsi_model = models.LsiModel(corpus=tfidf_vec, id2word=dictionary, num_topics=1000)

    print('Calculating TFIDF')
    for i in tqdm(range(len(texts))):
        string_lsi = lsi_model[dictionary.doc2bow(texts[i])]
        output_emb = np.asarray(string_lsi)[: , 1].reshape([1, -1])
        go_embeddings[go_classes[i]] = output_emb
    emb_dim = np.size(output_emb, 1)
    print('TFIDF Embedding Dim: {}'.format(emb_dim))
    save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def get_clusDCA_emb(cfg):
    '''
    Get the graph embeddings of GO terms, we diredctly used the clusDCA codes here
    :param cfg:
    :return:
    '''
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())

    G = nx.Graph()
    G_distance = np.zeros([len(go_classes), len(go_classes)])
    G_distance = pd.DataFrame(G_distance)
    G_distance.columns, G_distance.index = go_classes, go_classes
    for node in go_classes:
        G.add_node(node)
    for node in tqdm(go_classes):
        for a in go_data[node]['is_a']:
            if a in go_classes:
                G.add_edge(node, a)
    print('Computing Graph Distance!')
    for node in tqdm(go_classes):
        length = nx.single_source_dijkstra_path_length(G, node)
        lKeys, length = list(length.keys()), list(length.values())
        G_distance.loc[node, lKeys] = length
    adjacent_GO = np.asarray(G_distance)
    sp = RandomWalkRestart(adjacent_GO, 0.8)
    network_npy = DCA_vector(sp, dim=500)[0]
    rank_e = np.linalg.matrix_rank(network_npy)
    print('Rank of your embeddings is {}'.format(rank_e))

    for i in tqdm(range(len(go_classes))):
        go_embeddings[go_classes[i]] = network_npy[i, :].reshape([1, -1])
    save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def get_Word2Vec_emb(cfg):
    '''
    Use the Word2Vec method to embed the textual descriptions of the Gene Ontology terms
    :param cfg:
    :return:
    '''
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())

    documents, sentences = [], []
    for t_id in go_classes:
        text_t = go_data[t_id]['name'] + '. ' + go_data[t_id]['def'][0]
        sentences.append(text_t)
        documents.append(word_tokenize(text_t))

    vec_size = 1000
    word2vec_model = Word2Vec(vector_size=vec_size, min_count=1, workers=32, alpha=0.1, min_alpha=0.025)
    word2vec_model.build_vocab(documents)
    word2vec_model.train(documents, total_examples=word2vec_model.corpus_count, epochs=500)
    for t_id in tqdm(go_classes):
        text_t = go_data[t_id]['name'] + '. ' + go_data[t_id]['def'][0]
        text_t = word_tokenize(text_t)
        emb_i = np.zeros([vec_size])
        scale = 0
        for word in text_t:
            scale += 1
            emb_i += word2vec_model.wv[word]
        emb_i = emb_i / scale
        go_embeddings[t_id] = emb_i.reshape([1, -1])
    print('Word2Vec embeddings finished!')
    save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def get_Doc2Vec_emb(cfg):
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())

    documents = []
    for t_id in go_classes:
        text_t = go_data[t_id]['name'] + '. ' + go_data[t_id]['def'][0]
        documents.append(word_tokenize(text_t))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    doc2vec_model = Doc2Vec(vector_size=1000, window=4, min_count=1, workers=32, alpha=0.1, min_alpha=0.025)
    doc2vec_model.build_vocab(documents)
    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=500)
    for t_id in go_classes:
        text_t = go_data[t_id]['name'] + '. ' + go_data[t_id]['def'][0]
        emb_i = doc2vec_model.infer_vector(word_tokenize(text_t))
        go_embeddings[t_id] = emb_i.reshape([1, -1])
    print('Word2Vec embeddings finished!')
    save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


def term2preds_label(preds, label, terms, terms2id):
    new_preds, new_label = [], []
    for t_id in terms:
        new_preds.append(preds[:, terms2id[t_id]].reshape((-1, 1)))
        new_label.append(label[:, terms2id[t_id]].reshape((-1, 1)))
    new_preds = np.concatenate(new_preds, axis=1)
    new_label = np.concatenate(new_label, axis=1)
    return new_preds, new_label


def organize_workingspace(workingspace):
    '''
    Make sure that the working space include the zero shot folder, few shot folder,
    model folder, training log folder and results folder
    :param workingspace:
    :return:
    '''
    task_path = workingspace
    model_path = task_path + '/model'
    logger_path = task_path + '/log'
    results_path = task_path + '/results'
    if not os.path.exists(workingspace):
        os.mkdir(workingspace)
        print('Warning: We created the working space: {}'.format(workingspace))
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    if not os.path.exists(results_path):
        os.mkdir(results_path)


def get_namespace_terms(terms, go, namespace):
    # select terms in the namespace
    name_dict = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    new_terms = []
    for t_id in terms:
        if go[t_id]['namespace'] == name_dict[namespace]:
            new_terms.append(t_id)
    return new_terms


def get_few_shot_namespace_terms(term_count, go, namespace, n=10):
    # select terms in namespace with less than n training sample
    name_dict = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    new_terms = []
    for t_id in term_count.keys():
        if go[t_id]['namespace'] == name_dict[namespace]:
            if term_count[t_id] <= n:
                new_terms.append(t_id)
    return new_terms


def term_training_numbers(val_data, train_data):
    '''
    This function calculates how many training samples exists in the
    :param val_data:
    :param train_data:
    :return:
    '''
    term2number = collections.OrderedDict()
    for annt_val in list(val_data['annotations']):
        for a_id in annt_val:
            term2number[a_id] = 0
    training_annt = list(train_data['annotations'])
    for annt in training_annt:
        annt = list(set(annt))
        for a_id in annt:
            if a_id not in term2number.keys():
                continue
            term2number[a_id] += 1
    return term2number


def compute_blast_preds(diamond_scores,
                        test,
                        train_data):
    '''
    This function computes the predictions of blast results.
    The codes here are borrowed from the DeepGOPlus paper
    :param diamond_scores:
    :param test:
    :param train_data:
    :param is_load:
    :param save_path:
    :return:
    '''
    blast_preds = collections.OrderedDict()
    print('Diamond preds')
    annotation_dict = collections.OrderedDict()
    for i in train_data.index:
        prot_id = train_data.loc[i]['proteins']
        annts = set(train_data.loc[i]['annotations'])
        if prot_id not in annotation_dict.keys():
            annotation_dict[prot_id] = collections.OrderedDict()
            for ann_id in annts:
                annotation_dict[prot_id][ann_id] = None
        else:
            for ann_id in annts:
                annotation_dict[prot_id][ann_id] = None

    for prot_id in tqdm(test.proteins):
        annots = {}

        # BlastKNN
        sim = collections.OrderedDict()
        if prot_id in diamond_scores.keys():
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= set(annotation_dict[p_id].keys())
                total_score += score
            allgos = set(allgos)
            for go_id in allgos:
                s = 0.0
                for p_id in sim_prots.keys():
                    score = sim_prots[p_id]
                    if go_id in annotation_dict[p_id].keys():
                        s += score
                sim[go_id] = s / total_score
        blast_preds[prot_id] = sim
    return blast_preds


def extract_terms_from_dataset(dataset):
    id = 0
    terms2id = collections.OrderedDict()
    id2terms = collections.OrderedDict()
    for i in dataset.index:
        annts = dataset.iloc[i]['annotations']
        for annt in annts:
            if annt not in terms2id:
                terms2id[annt] = id
                id2terms[id] = annt
                id += 1
    return terms2id, id2terms


def coef_vec(A, B):
    coefs_array = []
    for i in range(np.size(A, 0)):
        a1 = A[i, :]
        b1 = B[i, :]
        coefs = np.corrcoef(a1, b1)[0, 1]
        coefs_array.append(coefs)
    return np.asarray(coefs_array)


def sample_edges(st=0, ed=1000, n=100, exclude_edges=[]):
    samples = []
    while True:
        nodes = np.arange(st, ed)
        np.random.shuffle(nodes)
        edge = list(nodes[0: 2])
        if edge not in exclude_edges and edge not in samples:
            samples.append(edge)
        if len(samples) == n:
            break
    return np.asarray(samples)


def edge_probability(encodings_i, encodings_j, text_encodings_p):
    s1 = np.dot(encodings_i, np.transpose(text_encodings_p))
    s1 = 1 / (1 + np.exp(-s1))
    s2 = np.dot(encodings_j, np.transpose(text_encodings_p))
    s2 = 1 / (1 + np.exp(-s2))
    s3 = coef_vec(encodings_i, encodings_j)
    score = s3 * (0.5 * s1.squeeze() + 0.5 * s2.squeeze())
    return score