import random
import torch.nn as nn
from module.gt_mi import GT_MI_NCE
from module import Graph_Encoder
from module import LogReg
import torch.nn.functional as F


class SeFsnc(nn.Module):
    def __init__(self, num_feature, num_hidden, num_head, num_layers_gat, feat_drop, attn_drop, negative_slope,
                 mi_hidden, num_class, tree_height, tree_layer_weight, tree_drop, con_tau, big, batch):
        super(SeFsnc, self).__init__()
        heads = ([num_head] * num_layers_gat)
        self.graph_encoder = Graph_Encoder(num_layers_gat, num_feature, num_hidden, heads, F.elu, feat_drop, attn_drop,
                                           negative_slope)
        self.gt_mi = GT_MI_NCE(num_head, num_hidden, mi_hidden, con_tau, tree_height, tree_layer_weight, tree_drop, big,
                               batch)
        self.cls = LogReg(num_head * num_hidden, num_class)

    def get_graph_embed(self, graph, feature):
        graph_embed = self.graph_encoder(graph, feature)
        return graph_embed


    # tree
    def get_mi_loss(self, graph_embed, tree, num_head, contrast_weight):
        anchor_index = random.randint(0, num_head - 1)
        gt_mi_loss = self.gt_mi(graph_embed, anchor_index, tree, contrast_weight)
        return gt_mi_loss

    def get_logits(self, graph_embed, cls_weight=None):
        logits = self.cls(graph_embed, cls_weight)
        return logits




