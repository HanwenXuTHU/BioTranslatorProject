import collections
import numpy as np
import logging
import pickle
import pandas as pd
import scipy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def sentences_evaluation(preds, labels):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_score = collections.OrderedDict()
    rouge_score = collections.OrderedDict()
    all_metrics = collections.OrderedDict()
    meteor, nist = [], []
    for i in range(1, 5, 1):
        weights = [0, 0, 0, 0]
        weights[i - 1] = 1
        bleu_score['bleu{}'.format(i)] = []
        for j in range(len(preds)):
            bleu_score['bleu{}'.format(i)].append(sentence_bleu([labels[j]],
                                                                preds[j],
                                                                weights=weights))
        bleu_score['bleu{}_avg'.format(i)] = np.mean(bleu_score['bleu{}'.format(i)])

    bleu_score['bleu'] = []
    for j in range(len(preds)):
        bleu_score['bleu'].append(sentence_bleu([labels[j]], preds[j]))
    bleu_score['bleu_avg'] = np.mean(bleu_score['bleu'])

    for i in [1, 2, 'L']:
        rouge_score['rouge{}'.format(i)] = []
        for j in range(len(preds)):
            score = scorer.score(labels[j], preds[j])
            rouge_score['rouge{}'.format(i)].append(score['rouge{}'.format(i)])
        rouge_score['rouge{}_avg'.format(i)] = np.mean(rouge_score['rouge{}'.format(i)])

    for j in range(len(preds)):
        meteor.append(meteor_score([labels[j]], preds[j]))
        nist.append(sentence_nist([labels[j]], preds[j]))

    all_metrics['bleu'], all_metrics['rouge'] = bleu_score, rouge_score
    all_metrics['meteor'], all_metrics['nist'] = np.mean(meteor), np.mean(nist)
    return bleu_score, rouge_score, np.mean(meteor), np.mean(nist), all_metrics


def get_logger(log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = log_name
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


def save_test_text(preds, nst_preds, labels, save_path):
    save_text = collections.OrderedDict()
    save_text['truth'] = labels
    save_text['preds'] = preds
    save_text['nearest'] = nst_preds
    save_text = pd.DataFrame(save_text)
    save_text.to_csv(save_path, index=False)


def calculate_jaccard(setA, setB):
    intersects = setA.intersection(setB)
    union = setA.union(setB)
    return len(intersects)/len(union)


def calculate_fisher(setA, setB, setN=1000):
    intersects = setA.intersection(setB)
    union = setA.union(setB)
    a, b = len(intersects), len(setB) - len(intersects)
    c, d = len(setA) - len(intersects), setN - len(setB) - len(setA) + len(intersects)
    _, pvalue = scipy.stats.fisher_exact([[a, b], [c, d]])
    return pvalue

