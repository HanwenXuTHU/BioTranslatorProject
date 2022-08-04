import click as ck
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from torch import squeeze
import torch
import random
import collections
from tqdm import tqdm
import pickle
from copy import deepcopy
from fairseq.data.dictionary import Dictionary
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import os
from fairseq_cli import preprocess
from argparse import ArgumentParser
import sys
import subprocess
import fairseq.options
import random


generate_path = '../data/goa_human_terms_10_proteins_fairseq'
parser = fairseq.options.get_preprocessing_parser()
parser.set_defaults(source_lang='en',
                    target_lang='de',
                    fp16=True,
                    workers=24,
                    trainpref=generate_path + '/train_fold',
                    validpref=generate_path + '/valid_fold',
                    testpref=generate_path + '/valid_fold',
                    destdir=generate_path + '/databin_fold')
args = parser.parse_args()
preprocess.main(args)


