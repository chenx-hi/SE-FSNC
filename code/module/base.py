import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

import numpy as np

from dgl.nn.pytorch import GATConv



class Graph_Encoder(nn.Module):
    def __init__(self, num_layers, num_feature, num_hidden, heads, activation, feat_drop, attn_drop, negative_slope):
        super(Graph_Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(num_feature, num_hidden, heads[0],
                                       feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, graph, inputs):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h = self.gat_layers[l](graph, temp)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        return heads


class Tree_Encoder(nn.Module):
    def __init__(self, dropout, big):
        super(Tree_Encoder, self).__init__()
        self.dropout = dropout
        self.big = big

    def forward(self, feat, tree_partitions):
        if self.big:
            tree_embed = []
        else:
            tree_embed = [feat]
        for k in range(len(tree_partitions) - 1):
            feat = F.dropout(feat, self.dropout, training=self.training)
            if k == len(tree_partitions) - 2:
                num_partition = feat.size(0)
                idx = torch.randperm(num_partition)
                shuf_feat = torch.spmm(tree_partitions[k].transpose(0, 1), feat[idx])
                feat = torch.spmm(tree_partitions[k].transpose(0, 1), feat)
                tree_embed.append(feat)
                tree_embed.append(shuf_feat)
            else:
                feat = torch.spmm(tree_partitions[k].transpose(0, 1), feat)
                tree_embed.append(feat)
        return tree_embed


class LogReg(nn.Module):
    def __init__(self, in_channel: int, num_class: int):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_channel, num_class)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, w):
        if w is None:
            return F.log_softmax(self.fc(seq), dim=1)
        else:
            return F.log_softmax(F.linear(seq, w[0], w[1]), dim=1)
