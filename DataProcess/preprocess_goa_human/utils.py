import os
import torch
import shutil
import random
import pickle
import numpy as np
import collections
import pandas as pd
from collections import deque


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


def get_children(go, term_id):
    if term_id not in go:
        return set()
    term_set = set()
    q = deque()
    q.append(term_id)
    while(len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.add(t_id)
            for child_id in go[t_id]['children']:
                if child_id in go:
                    q.append(child_id)
    return term_set


def propagate_annotations(annotations, go):
    ann_prop = []
    for t_i in annotations:
        ann_prop += get_anchestors(go, t_i)
    ann_prop = list(set(ann_prop))
    return ann_prop


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def mycopyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)
        print ("copy %s -> %s"%(srcfile, dstpath + fname))


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def get_description_embedding(text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to('cuda')
        outputs = model(inputs.data['input_ids'][:, 0: min(512, inputs.data['input_ids'].shape[1])])
        second_to_last = torch.mean(outputs.last_hidden_state, 1)
        embedding = np.asarray(second_to_last.cpu()).reshape([1, -1])
    return embedding


def k_fold_split(train_df, k, seed=123):
    train_num = train_df.index.size
    idx = np.arange(train_num)
    np.random.seed(seed)
    np.random.shuffle(idx)
    fold_size = int(train_num / k)
    fold_valid, fold_idx, fold_training = [], [], []
    for i in range(k):
        fold_idx.append(idx[i*fold_size : (i + 1)*fold_size])
    for i in range(k):
        fold_valid.append(train_df.iloc[fold_idx[i]])
        training_set, is_append = [], 0
        for j in range(k):
            if j != i:
                training_set.append(train_df.iloc[fold_idx[j]])
        training_set = pd.concat(training_set)
        fold_training.append(training_set)
    return fold_valid, fold_training


def zero_shot_terms(fold_val, go, term_list, k=3, r=1):
    go_terms = go
    fold_zero_shot_terms = []
    term_length = len(term_list)
    leaf_node = []
    for i in range(term_length):
        term_i = term_list[i]
        children_i = get_children(go_terms, term_i)
        if len(children_i) == 1 and term_i not in leaf_node: leaf_node.append(term_i)
    inference_num = int(r*len(leaf_node))

    for i in range(k):
        validation_set = fold_val[i]
        validation_terms = collections.OrderedDict()
        zero_shot_terms = collections.OrderedDict()
        random.shuffle(leaf_node)
        for annt_val in list(validation_set['annotations']):
            for a_id in annt_val:
                validation_terms[a_id] = 0
        for t_id in leaf_node:
            if t_id in validation_terms.keys():
                zero_shot_terms[t_id] = 0
            if len(zero_shot_terms.keys()) >= inference_num:
                break
        fold_zero_shot_terms.append(zero_shot_terms)
    return fold_zero_shot_terms