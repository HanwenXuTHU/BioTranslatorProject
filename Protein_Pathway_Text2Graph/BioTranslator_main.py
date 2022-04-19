import torch
import torch.nn as nn
from model import BioTranslatorModel
from options import model_config, data_loading
from file_loader import FileLoader
from torch.utils.data import DataLoader
from utils import init_weights, EarlyStopping, compute_roc, vec2classes, update_model_config
import utils
from tqdm import tqdm
import numpy as np
import collections


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
    torch.cuda.set_device('cuda:' + data_opt.gpu_ids)
    model_opt = model_config()
    file = FileLoader(data_opt)
    model_opt = update_model_config(model_opt, file)
    model_predict = BioTranslator(model_opt)
    train_dataset = DataLoader(file.train_data, batch_size=model_opt.batch_size, shuffle=True)
    for i in range(model_opt.epoch):
        train_loss, inference_loss = 0, 0
        num = 0
        print('Training iters')
        for j, train_D in tqdm(enumerate(train_dataset)):
            model_predict.backward_model(train_D['seq'], train_D['description'], train_D['vector'], file.emb_tensor_train, train_D['label'])
            train_loss += float(model_predict.loss.item())
            num += 1
        train_loss = train_loss/num
        print('iter:{} train loss:{}'.format(i, train_loss))
        torch.save(model_predict, model_opt.save_path.format(data_opt.dataset))


if __name__ == '__main__':
    main()