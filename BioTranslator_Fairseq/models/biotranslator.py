# -*- coding: utf-8 -*-
import torch
import collections
import itertools
import numpy as np
from torch import nn
from torch.nn import init
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture


def init_weights(net, init_type='normal', init_gain=0.02):
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
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class SeqEncoder(nn.Module):
    def __init__(self, params):
        '''

        :param params: params[0] the input channel, params[1] the sequence length, params[2] output dim
        '''
        super(SeqEncoder, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, 129, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=params[0], out_channels=512, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=params[1] - kernels[i] + 1, stride=1)".format(i))
        self.seq_layer = [nn.Linear(len(kernels) * 512, params[2]), nn.LeakyReLU(inplace=True)]
        self.seq_layer = nn.Sequential(*self.seq_layer)

    def forward(self, seq):
        emb_list = []
        for i in range(self.kernel_num):
            exec("seq_i = self.conv1d_{}(seq)".format(i))
            exec("seq_i = self.pool1d_{}(seq_i)".format(i))
            exec("emb_list.append(torch.squeeze(seq_i).reshape([seq.size(0), -1]))")
        output = self.seq_layer(torch.cat(tuple(emb_list), dim=1))
        return output


class VecEncoder(nn.Module):

    def __init__(self, params):
        super(VecEncoder, self).__init__()
        self.vec_layer = [nn.Linear(params[0], params[1]), nn.LeakyReLU(inplace=True)]
        self.vec_layer = nn.Sequential(*self.vec_layer)

    def forward(self, vec):
        return self.vec_layer(vec)


# BioTranslator Model
class BioDataTranslator(nn.Module):

    def __init__(self, task, text_dim):

        super(BioDataTranslator, self).__init__()
        self.data_types = task.data_types.split('_')
        self.cat_dim = 0
        for tp in self.data_types:
            if 'seq' in tp:
                self.add_module(tp, SeqEncoder(task.model_dims[tp]))
                self.cat_dim += task.model_dims[tp][-1]
            if 'vec' in tp:
                self.add_module(tp, VecEncoder(task.model_dims[tp]))
                self.cat_dim += task.model_dims[tp][-1]
        self.cat2emb = nn.Linear(self.cat_dim, text_dim)

    def forward(self, input_list):
        for i in range(len(self.data_types)):
            if i == 0:
                x_enc = self.get_submodule(self.data_types[i])(input_list[i])
            else:
                x_enc = torch.cat((x_enc, self.get_submodule(self.data_types[i])(input_list[i])), dim=1)
        return self.cat2emb(x_enc)


@register_model('biotranslator')
class BioDataAnnotator(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--text_dim', type=int, default=768)

    @classmethod
    def build_model(cls, args, task):
        data_translator = BioDataTranslator(task, args.text_dim)
        init_weights(data_translator, init_type='xavier')
        return cls(data_translator)

    def __init__(self, data_translator):
        super(BioDataAnnotator, self).__init__()
        self.data_translator = data_translator
        self.activation = torch.nn.Sigmoid()

    def forward(self, input_list, texts):
        # get textual description encodings
        text_encodings = texts.permute(1, 0)
        # get biology instance encodings
        data_encodings = self.data_translator(input_list)
        # compute logits
        logits = torch.matmul(data_encodings, text_encodings)
        return self.activation(logits)


@register_model_architecture('biotranslator', 'biotranslator_arch')
def annotator_architecture(args):
    args.text_dim = getattr(args, 'text_dim', 768)