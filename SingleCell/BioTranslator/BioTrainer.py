import os
import torch
import collections
import numpy as np
from tqdm import tqdm
from .BioConfig import BioConfig
from .BioLoader import BioLoader
from .BioModel import BioTranslator
from .BioMetrics import auroc_metrics
from torch.utils.data import DataLoader
from .BioUtils import save_obj, load_obj, get_logger
from .BioUtils import evaluate_auroc, evaluate_unseen_auroc, evaluate_auprc, evaluate_unseen_auprc


class BioTrainer:

    def __init__(self):
        '''
        This class mainly include the training of BioTranslator Model
        :param files:
        '''

    def setup_model(self, files: BioLoader, cfg: BioConfig):
        if cfg.method == 'BioTranslator':
            cfg.expr_dim = files.ngene
            self.model = BioTranslator(cfg)
        else:
            print('Warnings: No such method to setup the model!')

    def setup_training(self, cfg: BioConfig):
        # train epochs
        self.epoch = cfg.epoch
        # loss function
        self.loss_func = torch.nn.BCELoss()
        # use Adam as optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def backward_model(self, input_expr, emb_tensor, label):
        logits = self.model(input_expr=input_expr, texts=emb_tensor)
        self.loss = self.loss_func(logits, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def train(self, files: BioLoader, cfg: BioConfig):
        # store the results of each fold in one list
        results_cache = collections.OrderedDict()
        results_var = ['unseen_auroc', 'auroc', 'unseen_auprc', 'auprc']
        if cfg.task == 'same_dataset':
            for var in results_var:
                results_cache[var] = collections.OrderedDict()
                for unseen_ratio in cfg.unseen_ratio:
                    results_cache[var][unseen_ratio] = []
        # record the performance of our model
        self.logger = get_logger(cfg.logger_name)
        for unseen_ratio in cfg.unseen_ratio:
            for fold_i in range(cfg.n_iter):
                if cfg.task == 'same_dataset':
                    print('Start Cross-Validation Fold :{} Unseen Ratio: {}...'.format(fold_i, unseen_ratio))
                    files.generate_data(fold_i, unseen_ratio)
                elif cfg.task == 'cross_dataset':
                    print('Start Cross-Dataset Train :{} Validation: {}...'.format(cfg.dataset, cfg.eval_dataset))
                    files.generate_data()

                train_loader = DataLoader(files.train_data, batch_size=cfg.batch_size, shuffle=True)
                eval_loader = DataLoader(files.test_data, batch_size=cfg.batch_size)

                # setup the model architecture according the methods
                self.setup_model(files, cfg)
                # setup dataset, the training loss and optimizer
                self.setup_training(cfg)

                # train the model
                pbar = tqdm(range(self.epoch), desc='Training')
                for epoch_i in pbar:
                    train_loss = 0
                    train_count = 0
                    for batch in train_loader:
                        train_loss += self.backward_model(batch['features'], files.train_emb, batch['label'])
                        train_count += len(batch)
                        pbar.set_postfix({"Fold": fold_i,
                                          "epoch": epoch_i,
                                          "train loss": train_loss / train_count})

                # evaluate the model
                print('Evaluate Our Model ...')
                eval_loss, eval_count = 0, 0
                logits, label = [], []
                with torch.no_grad():
                    for batch in eval_loader:
                        pred_batch = self.model(input_expr=batch['features'], texts=files.test_emb)
                        eval_loss += self.loss_func(pred_batch, batch['label']).item()
                        eval_count += len(batch)
                        self.loss_func.zero_grad()
                        logits.append(pred_batch.cpu().numpy())
                        label.append(batch['label'].cpu().float().numpy())
                logits, label = np.concatenate(logits, axis=0), np.concatenate(label, axis=0)

                # evaluate the model on the zero shot tasks or the few shot tasks
                if cfg.task == 'same_dataset':
                    unseen_auroc = evaluate_unseen_auroc(logits, label, files.unseen2i)
                    auroc = evaluate_auroc(logits, label)
                    unseen_auprc = evaluate_unseen_auprc(logits, label, files.unseen2i)
                    auprc = evaluate_auprc(logits, label)
                    self.logger.info('Test all AUROCs :{} unseen AUROCs:{} AUPRCs :{} unseen AUPRCs:{}'.format(auroc, unseen_auroc,
                                                                                                    auprc,
                                                                                                    unseen_auprc))
                    results_cache['auroc'][unseen_ratio].append(auroc)
                    results_cache['unseen_auroc'][unseen_ratio].append(unseen_auroc)
                    results_cache['auprc'][unseen_ratio].append(auprc)
                    results_cache['unseen_auprc'][unseen_ratio].append(unseen_auprc)
                    torch.save(self.model, cfg.save_model_path.format(cfg.method, cfg.dataset, fold_i, unseen_ratio))

                if cfg.task == 'cross_dataset':
                    unseen_auroc = evaluate_unseen_auroc(logits, label, files.unseen2i)
                    auroc = evaluate_auroc(logits, label)
                    unseen_auprc = evaluate_unseen_auprc(logits, label, files.unseen2i)
                    auprc = evaluate_auprc(logits, label)
                    self.logger.info('Test all AUROCs :{} unseen AUROCs:{} AUPRCs :{} unseen AUPRCs:{}'.format(auroc, unseen_auroc,
                                                                                                    auprc,
                                                                                                    unseen_auprc))
                    results_cache['auroc'] = auroc
                    results_cache['unseen_auroc'] = unseen_auroc
                    results_cache['auprc'] = auprc
                    results_cache['unseen_auprc'] = unseen_auprc
                    torch.save(self.model, cfg.save_model_path.format(cfg.method, cfg.dataset, cfg.eval_dataset))

        # save the results
        save_obj(results_cache, cfg.results_name)







