import os
import torch
import logging
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from fairseq import utils
from torch import squeeze
from typing import Optional
from dataclasses import dataclass, field
import torchvision.transforms as transforms
from fairseq.dataclass import FairseqDataclass
from fairseq.data import FairseqDataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.logging import meters, metrics, progress_bar
from .protein_utils import (load_obj, get_BioTranslator_emb, emb2tensor,
                            load_go, compute_auroc_for_ont)
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils


logger = logging.getLogger(__name__)


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


class ProteinDataset(FairseqDataset):
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
        return {'proteins': self.p_ids[item],
                'prot_seq': squeeze(in_seq).float(),
                'prot_description': squeeze(description).float(),
                'prot_network': squeeze(vector).float(),
                'label': squeeze(label).float()}

    def collater(self, samples, pad_to_length=None):
        proteins = [s['proteins'] for s in samples]
        prot_seqs = [s['prot_seq'].unsqueeze(0) for s in samples]
        prot_descriptions = [s['prot_description'].unsqueeze(0) for s in samples]
        prot_networks = [s['prot_network'].unsqueeze(0) for s in samples]
        labels = [s['label'].unsqueeze(0) for s in samples]
        return {'proteins': proteins,
                'prot_seq': torch.cat(prot_seqs, dim=0),
                'prot_description': torch.cat(prot_descriptions, dim=0),
                'prot_network': torch.cat(prot_networks, dim=0),
                'label': torch.cat(labels, dim=0)}

    def __len__(self):
        return len(self.pSeq)

    def num_tokens(self, index):
        return len(self.p_ids[index].split(' '))

    def num_tokens_vec(self, indices):
        sizes = [len(self.p_ids[i].split(' ')) for i in indices]
        return np.asarray(sizes)

    def size(self, index):
        return len(self.p_ids[index])


@dataclass
class ProteinZeroShotConfig(FairseqDataclass):
    data: Optional[str] = field(
        default='/data/xuhw/data/ProteinDataset',
        metadata={
            "help": "The path of the protein dataset"
        },
    )

    data_name: Optional[str] = field(
        default='GOA_Human',
        metadata={
            "help": "The name of the protein dataset"
        },
    )

    emb_path: Optional[str] = field(
        default='/data/xuhw/data_emb/',
        metadata={
            "help": "The folder where you put the textual description embeddings"
        },
    )

    encoder_path: Optional[str] = field(
        default='../TextEncoder/Encoder/encoder.pth',
        metadata={
            "help": "The path of text encoder"
        },
    )

    k_fold: Optional[int] = field(default=3)
    gpu_ids: Optional[str] = field(default='0')


@register_task('protein_zero_shot', dataclass=ProteinZeroShotConfig)
class ProteinZeroShotTask(FairseqTask):

    def __init__(self, cfg: ProteinZeroShotConfig):
        super().__init__(cfg)
        self.gpu_ids = cfg.gpu_ids
        self.root_dir = cfg.data
        self.data_dir = os.path.join(cfg.data, cfg.data_name)
        # the key of labels in the data
        self.label_k = 'label'
        # read GO terms
        self.load_terms()
        # k fold
        self.k_fold = cfg.k_fold
        # load related data files
        self.load_related_data_files()
        # set the model params
        self.set_model_params()
        # load text embedding files
        self.load_text_emb(cfg)
        # load subset
        self.train_subset = 'train'
        self.valid_subset = 'valid'

    @classmethod
    def setup_task(cls, cfg: ProteinZeroShotConfig, **kwargs):
        return cls(cfg)

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def build_model(self, cfg):
        model = super().build_model(cfg)
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.optimizer.step()
        return loss, sample_size, logging_output

    def evaluate(self, cfg: ProteinZeroShotConfig, trainer, fold: int):
        zero_shot_terms = list(load_obj(os.path.join(self.data_dir,
                                                     'zero_shot_terms_fold_{}.pkl'.format(fold))))

        if cfg.dataset.fixed_validation_seed is not None:
            # set fixed seed for every validation
            utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

        trainer.begin_valid_epoch(0)
        subset = cfg.dataset.valid_subset
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=0,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        model = trainer.model
        criterion = trainer.criterion
        logits, prots, label = [], [], []
        for i, sample in enumerate(tqdm(progress)):
            if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                break
            with torch.no_grad():
                model.eval()
                criterion.eval()
                sample, _ = trainer._prepare_sample(sample)
                input_list = [sample[f] for f in self.features]
                pred_batch = model(input_list, self.train_emb)
                prots += sample['proteins']
                logits.append(pred_batch.cpu().numpy())
                label.append(sample['label'].cpu().float().numpy())
        logits, label = np.concatenate(logits, axis=0), np.concatenate(label, axis=0)
        results = compute_auroc_for_ont(zero_shot_terms, self.go_data, logits, label, self.terms2i)

        for ont in ['bp', 'mf', 'cc']:
            print('fold:{} ont: {} roc auc:{}'.format(fold, ont, results[ont]['auroc']))
            '''for T in results[ont]['percentage'].keys():
                print(
                    'fold:{} ont: {} the AUROC of {}% terms is greater than {}'
                        .format(fold, ont, results[ont]['percentage'][T], T))'''

    def load_terms(self):
        go_file = os.path.join(self.data_dir, 'go.obo')
        self.go_data = load_go(go_file)
        terms_file = os.path.join(self.data_dir, 'terms.pkl')
        terms = pd.read_pickle(terms_file)
        self.i2terms = list(terms['terms'])
        go_terms = list(self.go_data.keys())
        self.i2terms = list(set(self.i2terms).intersection(set(go_terms)))
        self.terms2i = collections.OrderedDict()
        self.n_classes = len(self.i2terms)
        print('terms number:{}'.format(len(self.i2terms)))
        for i in range(len(self.i2terms)): self.terms2i[self.i2terms[i]] = i

    def load_related_data_files(self):
        prot_network_file = os.path.join(self.data_dir, 'prot_network.pkl')
        prot_description_file = os.path.join(self.data_dir, 'prot_description.pkl')
        self.prot_network = load_obj(prot_network_file)
        self.network_dim = np.size(list(self.prot_network.values())[0])
        self.prot_description = load_obj(prot_description_file)

    def generate_fold_data(self, iter=None):
        dataset_file = os.path.join(self.data_dir, 'train_data_fold_{}.pkl'.format(iter))
        dataset = pd.read_pickle(dataset_file)
        self.train_data = self.zero_shot_data(dataset, iter)

        dataset_file = os.path.join(self.data_dir, 'validation_data_fold_{}.pkl'.format(iter))
        dataset = pd.read_pickle(dataset_file)
        self.test_data = dataset

        self.train_data = ProteinDataset(self.train_data,
                                          self.terms2i, self.prot_network,
                                          self.prot_description, gpu_ids=self.gpu_ids
                                          )
        self.test_data = ProteinDataset(self.test_data,
                                              self.terms2i, self.prot_network,
                                              self.prot_description, gpu_ids=self.gpu_ids
                                              )

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        self.datasets[self.train_subset] = self.train_data
        self.datasets[self.valid_subset] = self.test_data
        logger.info("Loaded {} with #samples: {}.".format(self.train_subset, len(self.train_data)))
        logger.info("Loaded {} with #samples: {}.".format(self.valid_subset, len(self.test_data)))
        return self.train_data

    def load_fold_data(self, k, train_fold_file, validation_fold_file):
        train_fold, val_fold = [], []
        for i in range(k):
            train_fold.append(pd.read_pickle(train_fold_file.format(i)))
            val_fold.append(pd.read_pickle(validation_fold_file.format(i)))
        return train_fold, val_fold

    def zero_shot_data(self, train_data, fold):
        zero_shot_terms = list(load_obj(os.path.join(self.data_dir,
                                                     'zero_shot_terms_fold_{}.pkl'.format(fold))))
        drop_index = []
        for j in train_data.index:
            annts = train_data.loc[j]['annotations']
            insct = list(set(annts).intersection(zero_shot_terms))
            if len(insct) > 0: drop_index.append(j)
        new_data = train_data.drop(index=drop_index)
        return new_data

    def set_model_params(self):
        # set input data types
        self.features = ['prot_seq', 'prot_description', 'prot_network']
        self.data_types = 'seq1_vec1_vec2'
        desc_size = np.size(self.prot_description[list(self.prot_description.keys())[0]])
        nwk_size = np.size(self.prot_network[list(self.prot_network.keys())[0]])
        self.model_dims = {'seq1': [21, 2000, 1500],
                           'vec1': [desc_size, 1500],
                           'vec2': [nwk_size, 1500]}

    def load_text_emb(self, cfg):
        if not os.path.exists(cfg.emb_path):
            os.mkdir(cfg.emb_path)
            print('Warning: We created the embedding folder: {}'.format(cfg.emb_path))

        if 'go_embeddings.pkl' not in os.listdir(cfg.emb_path):
            get_BioTranslator_emb(cfg, self.go_data)
        text_embedding_file = os.path.join(cfg.emb_path, 'go_embeddings.pkl')
        self.text_emb = pd.read_pickle(text_embedding_file)
        self.train_emb = emb2tensor(self.text_emb, self.terms2i)
        if len(cfg.gpu_ids) > 0:
            self.train_emb = self.train_emb.float().cuda()