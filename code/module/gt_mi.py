import torch
import torch.nn as nn

from . import Tree_Encoder
from .contrast import Contrast

class GT_MI_NCE(nn.Module):
    def __init__(self, num_head, num_hidden, mi_hidden, tau, tree_height, layer_weight, dropout, big, batch):
        super(GT_MI_NCE, self).__init__()
        self.tree_encoder = Tree_Encoder(dropout, big)
        ######## For GCN
        # self.proj = nn.Sequential(
        #     nn.Linear(num_hidden, mi_hidden),
        #     nn.ELU(),
        #     nn.Linear(mi_hidden, num_hidden))
        self.num_head = num_head
        self.con = Contrast(tau, tree_height, layer_weight, big, batch)
        self.big = big
        self.batch = batch

    def forward(self, graph_embed, anchor_index, tree, contrast_weight):
        tree_embed = self.tree_encoder(graph_embed[anchor_index], tree)
        gt_contrast_loss = 0
        for proj_head in graph_embed:
            gt_contrast_loss += self.con.get_mi(proj_head, tree_embed, contrast_weight)
        return gt_contrast_loss / self.num_head




