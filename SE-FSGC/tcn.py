#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gsp


class SEPooling(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size[[1,0]])
        return out
        # return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class TCN(torch.nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        self.args = args
        self.args.num_convs = args.tree_height
        self.args.conv = 'GAT'
        self.num_features = args.tree_fea_size#args.node_fea_size  # input_dim
        self.nhid = args.gin_hid  # hidden dim
        self.dropout_ratio = args.dropout
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()
        self.global_pool = gsp #if args.global_pooling == 'sum' else gap
        self.proj_head = nn.Sequential(
                nn.Linear(args.tree_height*self.nhid, self.nhid),
                nn.ReLU(inplace=True),
                nn.Linear(self.nhid, self.nhid))

    def __process_layer_edgeIndex(self, batch_data, layer=0):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            edge_mat_list.append(graph['graph_mats'][layer] + start_idx[i])
        edge_index = torch.cat(edge_mat_list, 1)
        return edge_index.to(self.args.device)

    def __process_sep_edgeIndex(self, batch_data, layer=1):
        edge_mat_list = []
        start_pdx = [0]
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            start_pdx.append(start_pdx[i] + graph['node_size'][layer-1])
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            edge_mat_list.append(torch.LongTensor(graph['edges'][layer])
                                 + torch.LongTensor([start_idx[i], start_pdx[i]]))
        edge_index = torch.cat(edge_mat_list, 0).T
        return edge_index.to(self.args.device)

    def __process_sep_size(self, batch_data, layer=1):
        size = [(graph['node_size'][layer-1], graph['node_size'][layer]) for graph in batch_data]
        return np.array(size).sum(axis=0)

    def __process_batch(self, batch_data, layer=0):
        batch = [[i] * graph['node_size'][layer] for i, graph in enumerate(batch_data)]
        batch = reduce(lambda x, y: x+y, batch)
        return torch.tensor(batch, dtype=torch.long).to(self.args.device)

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.num_convs):
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT':
                conv = GATConv(_input_dim, _output_dim, 2, concat=False)
            elif self.args.conv == 'Cheb':
                conv = ChebConv(_input_dim, _output_dim, K=2)
            elif self.args.conv == 'SAGE':
                conv = SAGEConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT2':
                conv = GATv2Conv(_input_dim, _output_dim, 2, concat=False)
            elif self.args.conv == 'Transformer':
                conv = TransformerConv(_input_dim, _output_dim, self.args.num_head, concat=False)
            elif self.args.conv == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        # nn.Linear(_output_dim, _output_dim),
                        # nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                    ), train_eps=False)
            convs.append(conv)
            _input_dim = _output_dim
        return convs

    def get_sepool(self):
        pools = nn.ModuleList()
        _input_dim = self.nhid
        _output_dim = self.nhid
        for _ in range(self.args.tree_height-1):
            pool = SEPooling(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ))
            pools.append(pool)
            _input_dim = _output_dim
        return pools


    def forward(self, batch_data):
        x = xIn = torch.cat([t['node_features'] for t in batch_data], dim=0).to(self.args.device)
        xs = []
        for _ in range(self.args.num_convs):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch_data, _)
            x = F.relu(self.convs[_](x, edge_index))
            # sep
            if _ < self.args.tree_height - 1:
                edge_index = self.__process_sep_edgeIndex(batch_data, _+1)
                size = self.__process_sep_size(batch_data, _+1)
                x = F.relu(self.sepools[_](x, edge_index, size=size))
            xs.append(x)

        pooled_xs = []
        # if self.args.link_input:
        #     batch = self.__process_batch(batch_data, 0)
        #     pooled_x = self.global_pool(xIn, batch)
        #     pooled_xs.append(pooled_x)
        for _, x in enumerate(xs):
            batch = self.__process_batch(batch_data, min(_+1, self.args.tree_height-1))
            pooled_x = self.global_pool(x, batch)
            pooled_xs.append(pooled_x)

        # For jumping knowledge scheme
        x = torch.cat(pooled_xs, dim=1)

        return self.proj_head(x)
