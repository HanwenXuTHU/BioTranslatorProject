import torch
from model import BioTranslatorModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes, compute_prc, update_model_config
from utils import compute_auroc_percentage, evaluate_model
import utils
from tqdm import tqdm
import numpy as np


class BioTranslator:

    def __init__(self, model_config):
        self.loss_func = torch.nn.BCELoss()
        self.model = BioTranslatorModel.BioTranslatorModel(input_nc=model_config.input_nc,
                                                in_nc=model_config.in_nc,
                                                max_kernels=model_config.max_kernels,
                                                hidden_dim=model_config.hidden_dim,
                                                feature=model_config.features,
                                                vector_dim=model_config.N_vector_dim,
                                                seqL=model_config.max_len,
                                                emb_dim=model_config.emb_dim)
        if len(model_config.gpu_ids) > 0:
            init_weights(self.model, init_type='xavier')
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_config.lr)

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        preds = self.model(input_seq, input_description, input_vector, emb_tensor)
        self.loss = self.loss_func(preds, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def main():
    data_opt = data_loading()
    model_opt = model_config()
    torch.cuda.set_device('cuda:' + data_opt.gpu_ids)
    file = FileLoader(data_opt)
    logger = utils.get_logger(data_opt.logger_name)
    for fold_i in range(3):
        model_opt = update_model_config(model_opt, file)
        model_predict = BioTranslator(model_opt)

        train_dataset = DataLoader(file.fold_training[fold_i], batch_size=model_opt.batch_size, shuffle=True)
        inference_dataset = DataLoader(file.fold_validation[fold_i], batch_size=model_opt.batch_size, shuffle=True)

        logger.info('fold : {}'.format(fold_i))
        for i in range(model_opt.epoch):
            print('Training iters')
            for j, train_D in tqdm(enumerate(train_dataset)):
                model_predict.backward_model(train_D['seq'], train_D['description'], train_D['vector'], file.emb_tensor_train, train_D['label'])

            inference_preds, inference_label, roc_auc = evaluate_model(model_predict, inference_dataset, file, fold_i)
            logger.info('fold:{} iter:{} zero-shot AUROC:{}'.format(fold_i, i, roc_auc))

            #compute auroc percentage:
            auroc_percentage = compute_auroc_percentage(inference_preds, inference_label, file, fold_i)
            logger.info('-------------------------------------------------------------')
            logger.info('| Fold | iter | AUROC greater than | Percentage |')
            for T in auroc_percentage.keys():
                logger.info('-------------------------------------------------------------')
                logger.info('| Fold  | iter  | AUROC greater than  | Percentage |')
                logger.info('|  {:^3}  |  {:^3}  |         {:.2f}         |     {:.2f}     |'.format(fold_i, i, T, auroc_percentage[T]))
        torch.save(model_predict, model_opt.save_path.format(fold_i))


if __name__ == '__main__':
    main()