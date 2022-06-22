# -*- coding: utf-8 -*-
import itertools
import numpy as np
from torch import nn
from torch.nn import init
from transformers import AutoTokenizer, AutoModel
import torch
import collections


# BioTranslator Model
class BioDataEncoder(nn.Module):

    def __init__(self,
                 feature=['seqs', 'network', 'description', 'expression'],
                 hidden_dim=1000,
                 seq_input_nc=4,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 expression_dim=1000,
                 drop_out=0.01,
                 text_dim=768):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param dense_num:
        :param seq_length:
        """
        super(BioDataEncoder, self).__init__()
        self.feature = feature
        self.text_dim = text_dim
        if 'seqs' in self.feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec("self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
                exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(i))
            self.fc_seq =[nn.Linear(len(kernels)*seq_in_nc, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_seq = nn.Sequential(*self.fc_seq)
        if 'description' in self.feature:
            self.fc_description = [nn.Linear(description_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_description = nn.Sequential(*self.fc_description)
        if 'network' in self.feature:
            self.fc_network = [nn.Linear(network_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_network = nn.Sequential(*self.fc_network)
        if 'expression' in self.feature:
            self.fc_expr = [nn.Linear(expression_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=drop_out)]
            self.fc_expr = nn.Sequential(*self.fc_expr)
        self.cat2emb = nn.Linear(len(self.feature)*hidden_dim, text_dim)

    def forward(self, x=None, x_description=None, x_vector=None, x_expr=None):
        x_list = []
        features = collections.OrderedDict()
        if 'seqs' in self.feature:
            for i in range(self.kernel_num):
                exec("x_i = self.conv1d_{}(x)".format(i))
                exec("x_i = self.pool1d_{}(x_i)".format(i))
                exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features['seqs'] = self.fc_seq(torch.cat(tuple(x_list), dim=1))
        if 'description' in self.feature:
            features['description'] = self.fc_description(x_description)
        if 'network' in self.feature:
            features['network'] = self.fc_network(x_vector)
        if 'expression' in self.feature:
            features['expression'] = self.fc_expr(x_expr)
        for i in range(len(self.feature)):
            if i == 0:
                x_enc = features[self.feature[0]]
            else:
                x_enc = torch.cat((x_enc, features[self.feature[i]]), dim=1)
        #x_enc = torch.nn.functional.normalize(x_cat, p=2, dim=1)
        return self.cat2emb(x_enc)


class BioTranslator(nn.Module):

    def __init__(self, cfg):
        super(BioTranslator, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.data_encoder = BioDataEncoder(feature=cfg.features,
                                           hidden_dim=cfg.hidden_dim,
                                           expression_dim=cfg.expr_dim,
                                           drop_out=cfg.drop_out,
                                           text_dim=cfg.term_enc_dim)
        self.activation = torch.nn.Softmax(dim=1)
        self.text_dim = cfg.term_enc_dim
        self.init_weights(self.data_encoder, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.data_encoder = self.data_encoder.to('cuda')
            self.activation = self.activation.to('cuda')

    def forward(self, input_expr, texts):
        # get textual description encodings
        text_encodings = texts.permute(1, 0)
        # get biology instance encodings
        data_encodings = self.data_encoder(x_expr=input_expr)
        # compute logits
        logits = torch.matmul(data_encodings, text_encodings)
        return self.activation(logits)

    def init_weights(self, net, init_type='normal', init_gain=0.02):
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
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)


# BaseModel
class BaseModel(nn.Module):

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.data_encoder = BioDataEncoder(feature=cfg.features,
                                           hidden_dim=cfg.hidden_dim,
                                           seq_input_nc=cfg.seq_input_nc,
                                           seq_in_nc=cfg.seq_in_nc,
                                           seq_max_kernels=cfg.seq_max_kernels,
                                           network_dim=cfg.network_dim,
                                           seq_length=cfg.max_length,
                                           text_dim=cfg.term_enc_dim)
        self.activation = torch.nn.Sigmoid()
        self.temperature = torch.tensor(0.07, requires_grad=True)
        self.text_dim = cfg.term_enc_dim
        self.init_weights(self.data_encoder, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.data_encoder = self.data_encoder.to('cuda')
            self.temperature = self.temperature.to('cuda')
            self.activation = self.activation.to('cuda')

    def forward(self, input_seq, input_description, input_vector, texts):
        # get textual description encodings
        text_encodings = texts.permute(1, 0)
        # get biology instance encodings
        data_encodings = self.data_encoder(input_seq, input_description, input_vector)
        # compute logits
        logits = torch.matmul(data_encodings, text_encodings)
        return self.activation(logits)

    def init_weights(self, net, init_type='normal', init_gain=0.02):
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
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)


# Base DataEncoder
class BaseDataEncoder(nn.Module):

    def __init__(self,
                 feature=['seqs', 'network', 'description', 'expression'],
                 hidden_dim=1000,
                 seq_input_nc=4,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 text_dim=768):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param dense_num:
        :param seq_length:
        """
        super(BaseDataEncoder, self).__init__()
        self.feature = feature
        self.text_dim = text_dim
        if 'seqs' in self.feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec("self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
                exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(i))
            self.fc_seq =[nn.Linear(len(kernels)*seq_in_nc, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_seq = nn.Sequential(*self.fc_seq)
        if 'description' in self.feature:
            self.fc_description = [nn.Linear(description_dim, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_description = nn.Sequential(*self.fc_description)
        if 'network' in self.feature:
            self.fc_network = [nn.Linear(network_dim, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_network = nn.Sequential(*self.fc_network)
        self.cat2emb = nn.Linear(len(self.feature)*hidden_dim, text_dim)

    def forward(self, x=None, x_description=None, x_vector=None):
        x_list = []
        features = collections.OrderedDict()
        if 'seqs' in self.feature:
            for i in range(self.kernel_num):
                exec("x_i = self.conv1d_{}(x)".format(i))
                exec("x_i = self.pool1d_{}(x_i)".format(i))
                exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features['seqs'] = self.fc_seq(torch.cat(tuple(x_list), dim=1))
        if 'description' in self.feature:
            features['description'] = self.fc_description(x_description)
        if 'network' in self.feature:
            features['network'] = self.fc_network(x_vector)
        for i in range(len(self.feature)):
            if i == 0:
                x_enc = features[self.feature[0]]
            else:
                x_enc = torch.cat((x_enc, features[self.feature[i]]), dim=1)
        #x_enc = torch.nn.functional.normalize(x_cat, p=2, dim=1)
        return self.cat2emb(x_enc)


# Our implementation of the DeepGOPlus Model
class DeepGOPlusModel(nn.Module):

    def __init__(self, input_nc=4, n_classes=1000, in_nc=512, max_kernels=129, seqL=2000):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(DeepGOPlusModel, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        self.fc.append(nn.Linear(len(kernels)*in_nc, n_classes))
        self.fc = nn.Sequential(*self.fc)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            exec("x_list.append(torch.squeeze(x_i))")
        x1 = torch.cat(tuple(x_list), dim=1)
        x2 = self.fc(x1)
        return self.sigmoid(x2)


class DeepGOPlus(nn.Module):
    '''
    Our implementation of the DeepGOPlus Model
    '''
    def __init__(self, cfg, n_classes=1000):
        super(DeepGOPlus, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.model = DeepGOPlusModel(input_nc=cfg.seq_input_nc,
                                            n_classes=n_classes,
                                            in_nc=cfg.seq_in_nc,
                                            max_kernels=cfg.seq_max_kernels,
                                            seqL=cfg.max_length)
        self.init_weights(self.model, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.model = self.model.to('cuda')

    def forward(self, input_seq, input_description, input_vector, emb_tensor):
        # Other features like input_description, input_vector and text embeddings will not be used
        # get biology instance annotations
        logits = self.model(input_seq)
        return logits

    def init_weights(self, net, init_type='normal', init_gain=0.02):
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
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

