# -*- coding: utf-8 -*-
"""
Created on Nov 15 15:55 2021

@author: Hanwen Xu

E-mail: xuhw20@mails.tsinghua.edu.cn

Pytorch version implementation of BioTranslator.
"""
from torch import nn
import torch


class BioTranslatorModel(nn.Module):

    def __init__(self,
                 ngene=2000,
                 nhidden=[1000],
                 ndim=100,
                 drop_out=0.1):
        super(BioTranslatorModel, self).__init__()
        self.ngene = ngene
        self.ndim = ndim
        self.nhidden = [self.ngene]
        self.nhidden.extend(nhidden)
        self.nhidden.append(self.ndim)
        self.nlayer = len(self.nhidden)
        self.fc = []
        for i in range(1, self.nlayer):
            self.fc.append(nn.Linear(self.nhidden[i - 1], self.nhidden[i]))
            if i != self.nlayer - 1:
                self.fc.append(nn.ReLU(inplace=True))
                self.fc.append(nn.Dropout(p=drop_out))
        self.fc_x = nn.Sequential(*self.fc)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x=None, emb_tensor=None):
        x1 = self.fc_x(x)
        emb2 = emb_tensor.permute(1, 0)
        x2 = torch.mm(x1, emb2)
        return self.activation(x2)