import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear, ModuleList, ReLU, Softplus
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool
import pdb


class GAT(nn.Module):

    def __init__(self,
                 x_dim=768,
                 graph_emb_dim=768,
                 num_layers=3,
                 GAT_heads=4,
                 additional_head=False):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        if (additional_head):
            self.additional_head = Sequential(Linear(x_dim, x_dim), ReLU())

        self.output_layer = Sequential(Linear(num_layers*graph_emb_dim, graph_emb_dim),
                                       ReLU())
        self.convs = ModuleList()
        act_dim = graph_emb_dim // GAT_heads
        for i in range(self.num_layers):
            if i:
                self.convs.append(GATConv(graph_emb_dim, act_dim, heads=GAT_heads, concat=True))
            else:
                self.convs.append(GATConv(x_dim, act_dim, heads=GAT_heads, concat=True))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, edge_attr):
        if self.additional_head:
            x1 = self.additional_head(x)

        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x1, edge_index))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        return embeddings_n


class MLP(nn.Module):

    def __init__(self, x_dim=768,
                 graph_emb_dim=768,
                 num_layers=3,
                 additional_head=False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        if (additional_head):
            self.additional_head = Sequential(Linear(x_dim, x_dim), ReLU())
        self.output_layer = Sequential(Linear(graph_emb_dim * num_layers, graph_emb_dim),
                                       ReLU(),
                                       Linear(graph_emb_dim, 768))
        self.convs = ModuleList()
        for i in range(self.num_layers):
            if (i):
                self.convs.append(Linear(graph_emb_dim, graph_emb_dim))
            else:
                self.convs.append(Linear(x_dim, graph_emb_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, edge_attr):

        if (self.additional_head):
            x = self.additional_head(x)

        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        return embeddings_n


class LinkPrediction(nn.Module):

    def __init__(self,
                 input_nc=4,
                 x_dim=768,
                 graph_emb_dim=768,
                 num_layers=3,
                 GAT_heads=4,
                 with_graph=True,
                 with_text=True,
                 max_kernels=129,
                 in_nc=512,
                 hidden_dim=[1000],
                 description_dim=768,
                 emb_dim=768,
                 seqL=200):
        super(LinkPrediction, self).__init__()
        self.with_text, self.with_graph = with_text, with_graph
        if with_graph:
            self.graphEncoder = GAT(x_dim=x_dim, graph_emb_dim=graph_emb_dim, num_layers=num_layers, GAT_heads=GAT_heads, additional_head=True)
        else:
            self.graphEncoder = MLP(x_dim=x_dim, graph_emb_dim=graph_emb_dim, num_layers=num_layers,
                                    additional_head=True)
        if with_text:
            self.projection_cat = Sequential(Linear(graph_emb_dim * 3, graph_emb_dim), ReLU(), Linear(graph_emb_dim, 2))
            self.projection_y = Sequential(Linear(graph_emb_dim, graph_emb_dim), nn.LeakyReLU(negative_slope=0.4))
        else:
            self.projection_cat = Sequential(Linear(2 * graph_emb_dim, 2))
        self.activation = torch.nn.Sigmoid()

    def forward(self, data):
        x1 = data.x

        edge_index = data.edge_index[:, data.train_edge_mask]
        edge_attr = data.edge_attr[data.train_edge_mask, :]
        emb = self.graphEncoder(x1, edge_index, data.batch, edge_attr)
        train_mask = data.train_edge_mask != True
        train_index = torch.tensor(range(train_mask.shape[0]), dtype=torch.long).to(train_mask.device)
        train_index = train_index[train_mask]
        pre_index_n = train_index.shape[0]
        sub_batch_n = pre_index_n // 500 + 1
        all_pred, all_std = [], []
        for batch_i in range(sub_batch_n):
            pre_batch_index = train_index[batch_i*500: (batch_i + 1)*500]

            pred_edge_index = data.edge_index[:, pre_batch_index]

            y = data.y
            y = y[data.batch[pred_edge_index[0]]]
            if self.with_text:
                y = self.projection_y(y)
                query = torch.cat([emb[pred_edge_index[0]], emb[pred_edge_index[1]], y], -1)
                query = self.projection_cat(query)
                pred = query
                #pred = self.projection_q(pred)
            else:
                query = torch.cat([emb[pred_edge_index[0]], emb[pred_edge_index[1]]], -1)
                query = self.projection_cat(query)
                pred = query
            pred = self.activation(pred)
            std = data.edge_label[pre_batch_index]
            all_pred.append(pred)
            all_std.append(std)
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        all_std = torch.cat(all_std, dim=0).squeeze()
        return all_pred, all_std