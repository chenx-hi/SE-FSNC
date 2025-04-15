import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Graph_Encoder, Tree_Encoder

class Encoder(nn.Module):
    def __init__(self, num_feature, num_hidden, num_heads, dropout, big):
        super(Encoder, self).__init__()
        self.g_encoder = Graph_Encoder(num_feature, num_hidden, num_heads, dropout)
        self.t_encoder = Tree_Encoder(dropout, big)

    def forward(self, feat, adj, tree_partitions, anchor_index):
        anchor_feat, heads = self.g_encoder(feat, adj, anchor_index)
        tree_embed = self.t_encoder(anchor_feat, tree_partitions)
        return tree_embed, heads

    def g_embed(self, feat, adj):
        _, heads = self.g_encoder(feat, adj, 0)
        return heads

