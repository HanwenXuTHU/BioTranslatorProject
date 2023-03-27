'''
To generate the GOA (Human) dataset, we need the following files:
(1) goa_human.gaf
(2) go.obo
(3) string_human_genes.txt
(4) string_human_mashup_vectors_d800.txt
(5) gene_alias_description.txt
(6) uniprot_sprot.fasta
'''
import os
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import load, save_obj, propagate_annotations, zero_shot_terms
from utils import get_description_embedding, k_fold_split, del_file, mycopyfile


# The path save the processed data
generate_repo = '/data/xuhw/data/ProteinDataset/GOA_Human/'
# The path containing necessary files for processing data
gaf_path = '../raw_data/goa_human.gaf'
go_path = '../raw_data/go.obo'
string_genes_path = '../raw_data/string_human_genes.txt'
string_network_path = '../raw_data/string_human_mashup_vectors_d800.txt'
gene_description_path = '../raw_data/gene_alias_description.txt'
uniprot_path = '../raw_data/uniprot_sprot.fasta'
stringid2gene_path = '../raw_data/9606.protein.aliases.v11.0.txt'
# use the following evidence
use_evidence = ['TAS', 'RCA', 'ND', 'NAS', 'ISS', 'IPI', 'IMP', 'IGI', 'IEP', 'IEA', 'IDA', 'IC']
# bert name
bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# k-fold cross-validation
k = 3


def main():
    # load gaf file
    # uniprot_id to go
    f = open(gaf_path)
    lines = f.readlines()
    goa = collections.OrderedDict()
    for line in tqdm(lines):
        if '!' in line:
            continue
        line = line.split('\t')
        prot_id, go_id, evi_id = line[1], line[4], line[6]
        if evi_id in use_evidence:
            if prot_id not in goa.keys():
                goa[prot_id] = []
            if go_id not in goa[prot_id]:
                goa[prot_id].append(go_id)
    f.close()

    # load go.obo file
    go = load(go_path)

    # load gene network vectors
    # gene_name to vector
    network = collections.OrderedDict()
    f1 = open(string_genes_path)
    lines1 = f1.readlines()
    f2 = open(string_network_path)
    lines2 = f2.readlines()
    for i in range(len(lines1)):
        gn = lines1[i].strip()
        network_vector = np.asarray(lines2[i].split(), dtype=np.float)
        network[gn] = network_vector
    f1.close()
    f2.close()

    # load gene description file
    # gene name to description
    description = collections.OrderedDict()
    f = open(gene_description_path)
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        description[line[0].strip()] = line[2].strip()
    f.close()

    # load uniprot file
    # uniprot_id to sequence
    # uniprot_id to gene_name
    uniprot = collections.OrderedDict()
    uniprotid2gn = collections.OrderedDict()
    f = open(uniprot_path)
    lines = f.readlines()
    uniprotid = None
    for line in tqdm(lines):
        if '>' in line:
            uniprotid = line.split('|')[1]
            if ' GN=' in line:
                gn = line.split(' GN=')[1].split(' ')[0]
            else:
                gn = 'Null'
            uniprot[uniprotid] = ''
            uniprotid2gn[uniprotid] = gn
        else:
            uniprot[uniprotid] += line.strip()
    f.close()

    # load mapping between the string id and gene name
    stringid2gn = collections.OrderedDict()
    f = open(stringid2gene_path)
    lines = f.readlines()
    for line in lines:
        if '## string_protein_id ## alias ## source ##' in line:
            continue
        stringid = line.split('\t')[0].split('9606.')[1]
        gn = line.split('\t')[1]
        stringid2gn[stringid] = gn
    f.close()

    # find uniprot_id intersections between different files
    # uniprot_id to gene_name
    intersect_uniprots = collections.OrderedDict()
    for uniprot_id_goa in tqdm(goa.keys()):
        if uniprot_id_goa in uniprot.keys():
            gn = uniprotid2gn[uniprot_id_goa]
            if gn not in network.keys() or gn not in description.keys() or gn in intersect_uniprots.values():
                continue
            intersect_uniprots[uniprot_id_goa] = gn

    # construct our dataset
    dataset = collections.OrderedDict()
    var_names = ['proteins', 'genes', 'sequences', 'annotations']
    for var in var_names:
        dataset[var] = []
    for uniprotid in intersect_uniprots:
        dataset['proteins'].append(uniprotid)
        dataset['genes'].append(intersect_uniprots[uniprotid])
        dataset['sequences'].append(uniprot[uniprotid])
        dataset['annotations'].append(propagate_annotations(goa[uniprotid], go))
    dataset = pd.DataFrame(dataset)

    # summarize term list
    terms = collections.OrderedDict()
    terms['terms'] = []
    for annt in dataset['annotations']:
        terms['terms'] += annt
    terms['terms'] = list(set(terms['terms']))
    terms = pd.DataFrame(terms)

    # convert gene_description to pubmedbert embeddings
    # uniprotid to gene_description_embeddings
    description_features = collections.OrderedDict()
    # load pubmedbert
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    model = AutoModel.from_pretrained(bert_name)
    model = model.to('cuda')
    model.eval()
    for uniprotid in tqdm(intersect_uniprots.keys()):
        text = description[intersect_uniprots[uniprotid]]
        description_features[uniprotid] = get_description_embedding(text, tokenizer, model)

    # extract network features
    # uniprotid to mashup vector
    network_features = collections.OrderedDict()
    for uniprotid in tqdm(intersect_uniprots.keys()):
        network_features[uniprotid] = network[intersect_uniprots[uniprotid]]

    # split dataset
    data_val, data_train = k_fold_split(dataset, k)

    # split zero shot terms
    fold_zero_shot_terms = zero_shot_terms(data_val, go, list(terms['terms']), k=k)

    # save dataset
    # refresh folder
    if os.path.exists(generate_repo):
        del_file(generate_repo)
    else:
        os.makedirs(generate_repo)
    # save all data
    save_obj(dataset, generate_repo + 'dataset.pkl')
    # save fold cross-validation and zero_shot_terms
    for i in range(k):
        save_obj(data_val[i], generate_repo + 'validation_data_fold_{}.pkl'.format(i))
        save_obj(data_train[i], generate_repo + 'train_data_fold_{}.pkl'.format(i))
        save_obj(fold_zero_shot_terms[i], generate_repo + 'zero_shot_terms_fold_{}.pkl'.format(i))
    # save terms
    save_obj(terms, generate_repo + 'terms.pkl')
    # save description_features
    save_obj(description_features, generate_repo + 'prot_description.pkl')
    # save network_features
    save_obj(network_features, generate_repo + 'prot_network.pkl')
    # copy go file to repo folder
    mycopyfile(go_path, generate_repo)
    print('Successfully generate dataset!')
    print('{} proteins finally detected!'.format(len(intersect_uniprots.keys())))


if __name__ == '__main__':
    main()