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
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from .single_cell_utils import (evaluate_unseen_auroc, evaluate_auroc,
                                evaluate_unseen_auprc, evaluate_auprc)
from .single_cell_utils import get_BioTranslator_emb, emb2tensor, SplitTrainTest, find_gene_ind
from .single_cell_utils import DataProcessing, read_data, read_data_file, read_ontology_file


logger = logging.getLogger(__name__)


class cellData(FairseqDataset):

    def __init__(self, expression_matrix, labels, nlabel=0):
        self.expression_matrix = expression_matrix
        self.labels = labels
        self.nlabel = nlabel

    def __getitem__(self, item):
        x = np.asarray(self.expression_matrix[item, :].todense()).squeeze()
        y = np.zeros(self.nlabel)
        y[self.labels[item]] = float(1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return {'expression': x, 'label': y}

    def __len__(self):
        return len(self.labels)

    def collater(self, samples, pad_to_length=None):
        features = [s['expression'].unsqueeze(0) for s in samples]
        labels = [s['label'].unsqueeze(0) for s in samples]
        return {'expression': torch.cat(features, dim=0),
                'label': torch.cat(labels, dim=0)}

    def num_tokens(self, index):
        return 1

    def num_tokens_vec(self, indices):
        sizes = [1 for i in indices]
        return np.asarray(sizes)

    def size(self, index):
        return 1


@dataclass
class CellTypeDiscoveryConfig(FairseqDataclass):

    data: Optional[str] = field(
        default='/data/xuhw/data/sc_data',
        metadata={
            "help": 'the path where you put the single cell dataset'
        },
    )

    data_name: Optional[str] = field(
        default='muris_droplet',
        metadata={
            "help": 'choose between sapiens, tabula_microcebus, '
                    'muris_droplet, microcebusAntoine, microcebusBernard, '
                    'microcebusMartine, microcebusStumpy, muris_facs'
        },
    )

    emb_path: Optional[str] = field(
        default='/data/xuhw/data_emb/',
        metadata={
            "help": "The folder where you put the textual description embeddings"
        },
    )

    ontology_repo: Optional[str] = field(
        default='/data',
        metadata={
            "help": "The folder where you put the ontology data"
        },
    )

    encoder_path: Optional[str] = field(
        default='../TextEncoder/Encoder/encoder.pth',
        metadata={
            "help": "The path of text encoder"
        },
    )

    backup_file: Optional[str] = field(
        default='backup_files/',
        metadata={
            "help": "where you backup files when using memory saving mode"
        },
    )

    unseen_ratio: Optional[float] = field(default=0.5)
    nfold_sample: Optional[float] = field(default=0.2)
    memory_saving_mode: Optional[bool] = field(default=True)
    k_fold: Optional[int] = field(default=5)
    gpu_ids: Optional[str] = field(default='0')


@register_task('cell_type_discovery', dataclass=CellTypeDiscoveryConfig)
class CellTypeDiscoveryTask(FairseqTask):

    def __init__(self, cfg: CellTypeDiscoveryConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.label_k = 'label'
        self.unseen_ratio = cfg.unseen_ratio
        self.load_text_emb(cfg)
        # load single cell data files
        self.load_files(cfg)
        # load subset
        self.train_subset = 'train'
        self.valid_subset = 'valid'

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
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def load_files(self, cfg):
        self.terms_with_def = self.text_emb_dict.keys()
        self.cell_type_nlp_emb_file, self.cell_type_network_file, self.cl_obo_file = read_ontology_file(
            'cell ontology', cfg.ontology_repo)
        self.DataProcess_obj = DataProcessing(self.cell_type_network_file, self.cell_type_nlp_emb_file, memory_saving_mode=cfg.memory_saving_mode, terms_with_def=self.terms_with_def)
        feature_file, filter_key, label_key, label_file, gene_file = read_data_file(cfg.data_name, cfg.data)
        self.feature, self.genes, self.label, _, _ = read_data(feature_file, cell_ontology_ids=self.DataProcess_obj.cell_ontology_ids,
                                                exclude_non_leaf_ontology=True, tissue_key='tissue',
                                                filter_key=filter_key, AnnData_label_key=label_key,
                                                nlp_mapping=False, cl_obo_file=self.cl_obo_file,
                                                cell_ontology_file=self.cell_type_network_file, co2emb=self.DataProcess_obj.co2vec_nlp,
                                                               memory_saving_mode=cfg.memory_saving_mode,
                                                               backup_file=cfg.backup_file)

    def load_text_emb(self, cfg):
        if 'co_embeddings.pkl' not in os.listdir(cfg.emb_path):
            get_BioTranslator_emb(cfg)
        text_embedding_file = cfg.emb_path + 'co_embeddings.pkl'
        self.text_emb_dict = pd.read_pickle(text_embedding_file)

    def generate_fold_data(self, iter=None):
        train_feature, train_label, test_feature, test_label = SplitTrainTest(self.feature,
                                                                              self.label,
                                                                              nfold_cls=self.unseen_ratio,
                                                                              nfold_sample=self.cfg.nfold_sample,
                                                                              random_state=iter,
                                                                              memory_saving_mode=self.cfg.memory_saving_mode)
        train_genes, test_genes = self.genes, self.genes
        self.DataProcess_obj.EmbedCellTypes(train_label)
        train_feature, test_feature, train_genes, test_genes = self.DataProcess_obj.ProcessTrainFeature(
            train_feature, train_label, train_genes, test_feature=test_feature, test_genes=test_genes,
            batch_correct=False)

        test_feature, mapping = self.DataProcess_obj.ProcessTestFeature(test_feature, test_genes, log_transform=False)

        self.unseen2i = collections.OrderedDict()
        self.train2i = collections.OrderedDict()
        self.test2i = collections.OrderedDict()
        idx = 0
        for x in set(train_label):
            self.train2i[x] = idx
            idx += 1

        idx = 0
        for x in set(test_label).union(set(train_label)):
            self.test2i[x] = idx
            if x not in set(train_label):
                self.unseen2i[x] = idx
            idx += 1

        train_label = [self.train2i[x] for x in train_label]
        test_label = [self.test2i[x] for x in test_label]

        self.train_data = cellData(train_feature, train_label, nlabel=len(self.train2i.keys()))
        self.test_data = cellData(test_feature, test_label, nlabel=len(self.test2i.keys()))

        self.train_emb = emb2tensor(self.text_emb_dict, self.train2i)
        if len(self.cfg.gpu_ids) > 0:
            self.train_emb = self.train_emb.cuda()
        self.test_emb = emb2tensor(self.text_emb_dict, self.test2i)
        if len(self.cfg.gpu_ids) > 0:
            self.test_emb = self.test_emb.cuda()

        self.train_genes = train_genes
        self.test_genes = test_genes
        self.ngene = len(train_genes)
        self.ndim = self.train_emb.size(1)
        self.set_model_params()

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        self.datasets[self.train_subset] = self.train_data
        self.datasets[self.valid_subset] = self.test_data
        logger.info("Loaded {} with #samples: {}.".format(self.train_subset, len(self.train_data)))
        logger.info("Loaded {} with #samples: {}.".format(self.valid_subset, len(self.test_data)))
        return self.train_data

    def set_model_params(self):
        # set input data types
        self.features = ['expression']
        self.data_types = 'vec1'
        self.model_dims = {'vec1': [self.ngene, 30]}

    def evaluate(self, cfg: CellTypeDiscoveryConfig, trainer, fold: int):
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
        logits, label = [], []
        for i, sample in enumerate(tqdm(progress)):
            with torch.no_grad():
                model.eval()
                criterion.eval()
                sample, _ = trainer._prepare_sample(sample)
                input_list = [sample[f] for f in self.features]
                pred_batch = model(input_list, self.test_emb)
                logits.append(pred_batch.cpu().numpy())
                label.append(sample['label'].cpu().float().numpy())
        logits, label = np.concatenate(logits, axis=0), np.concatenate(label, axis=0)

        unseen_auroc = evaluate_unseen_auroc(logits, label, self.unseen2i)
        auroc = evaluate_auroc(logits, label)
        unseen_auprc = evaluate_unseen_auprc(logits, label, self.unseen2i)
        auprc = evaluate_auprc(logits, label)
        print('Fold: {} Test all AUROCs :{} unseen AUROCs:{} AUPRCs :{} unseen AUPRCs:{}'.format(fold,
                                                                                                 auroc, unseen_auroc,
                                                                                                 auprc,
                                                                                                 unseen_auprc))