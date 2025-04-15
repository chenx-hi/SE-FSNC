import numpy as np
import torch
import torch.nn.functional as F


class Contrast:
    def __init__(self, tau, tree_height, layer_weight, big, batch, norm: bool = True):
        self.temperature = tau
        self.big = big
        self.batch = batch
        if self.big:
            self.tree_height = tree_height - 1
        else:
            self.tree_height = tree_height
        self.layer_weight = layer_weight
        self.norm = norm

    #
    def cosine_sim(self, z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
        if hidden_norm:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
        f = lambda x: torch.exp(x / self.temperature)
        return f(torch.mm(z1, z2.t()))


    def get_pos_mi(self, sim, weight):
        weight_index, weight_value = weight._indices(), weight._values()
        pos_mi = torch.zeros(sim.size(0)).cuda()
        pos_mi.index_add_(0, weight_index[0], weight_value * sim[weight_index[0], weight_index[1]])
        return pos_mi


    def get_mi(self, x, H, contrast_weight):
        gt_contrast_loss = 0
        contrast_weight_batch = []
        if self.big:
            idx = np.random.choice(x.shape[0], self.batch, replace=False)
            idx.sort()
            x = x[idx]
            idx_cuda = torch.tensor(idx.tolist()).to('cuda')
            for weight in contrast_weight:
                contrast_weight_batch.append(weight.index_select(0, idx_cuda))
        else:
            contrast_weight_batch = contrast_weight

        for i in range(0, self.tree_height):
            # Compute cosine similarity between nodes and communities
            cosine_sim = self.cosine_sim(x, H[i])
            pos_mi = self.get_pos_mi(cosine_sim, contrast_weight_batch[i])
            neg_mi = cosine_sim.sum(dim=1)

            if i == self.tree_height - 1:
                shuffle_cosine_sim = self.cosine_sim(x, H[-1])
                shuffle_mi = shuffle_cosine_sim.sum(dim=1)
                gt_info_nce = -torch.log(pos_mi / (neg_mi + shuffle_mi + 1e-8)).mean()
            else:
                gt_info_nce = -torch.log(pos_mi / (neg_mi + 1e-8)).mean()
            gt_contrast_loss += self.layer_weight[i] * gt_info_nce
        return gt_contrast_loss




