# -*- coding: utf-8 -*-
from torch import nn
import torch
import collections


class BioTranslatorModel(nn.Module):

    def __init__(self,
                 input_nc=4,
                 in_nc=512,
                 max_kernels=129,
                 seqL=2000,
                 hidden_dim=[1000],
                 feature=['seqs', 'network', 'protein description'],
                 vector_dim=800,
                 description_dim=768,
                 emb_dim=768):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(BioTranslatorModel, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, max_kernels, 8)
        self.kernel_num = len(kernels)
        self.feature = feature
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        self.hidden_dim = []
        self.hidden_dim.append(len(kernels)*in_nc)
        for i in range(len(hidden_dim)):
            self.hidden_dim.append(hidden_dim[i])
        for i in range(len(self.hidden_dim) - 1):
            self.fc.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.fc.append(nn.ReLU(inplace=True))
        self.fc_x = nn.Sequential(*self.fc)
        self.fc_vector = [nn.Linear(vector_dim, hidden_dim[-1]), nn.ReLU(inplace=True)]
        self.fc_description = [nn.Linear(description_dim, hidden_dim[-1]), nn.ReLU(inplace=True)]
        self.fc_vector = nn.Sequential(*self.fc_vector)
        self.fc_description = nn.Sequential(*self.fc_description)
        self.cat2emb = nn.Linear(len(self.feature)*hidden_dim[-1], emb_dim)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x=None, x_description=None, x_vector=None, emb_tensor=None):
        x_list = []
        features = collections.OrderedDict()
        if 'seqs' in self.feature:
            for i in range(self.kernel_num):
                exec("x_i = self.conv1d_{}(x)".format(i))
                exec("x_i = self.pool1d_{}(x_i)".format(i))
                exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features['seqs'] = self.fc_x(torch.cat(tuple(x_list), dim=1))
        if 'protein description' in self.feature:
            features['protein description'] = self.fc_description(x_description)
        if 'network' in self.feature:
            features['network'] = self.fc_vector(x_vector)
        for i in range(len(self.feature)):
            if i == 0:
                x_cat = features[self.feature[0]]
            else:
                x_cat = torch.cat((x_cat, features[self.feature[i]]), dim=1)
        x1 = self.cat2emb(x_cat)
        emb2 = emb_tensor.permute(1, 0)
        x2 = torch.mm(x1, emb2)
        return self.activation(x2)