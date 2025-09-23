import torch
import torch_sparse
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull
import random
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from .codingTree_utils import get_tree_partition
import dgl
import scipy.sparse as sp
import os


class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    'dblp': {"train": 77, 'dev': 30, 'test': 30},
    'Amazon-Clothing': {"train": 37, 'dev': 20, 'test': 20},
}

data_pth = "../dataset/"


class DataSet():
    def __init__(self, dataset, graph, x, y, num_class, tree_partition, contrast_weight, id_by_class, train_class,
                 val_class, test_class):
        self.dataset = dataset
        self.graph = graph
        self.x = x
        self.y = y
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = num_class
        self.tree_partition = tree_partition
        self.contrast_weight = contrast_weight
        self.id_by_class = id_by_class
        self.train_class = train_class
        self.val_class = val_class
        self.test_class = test_class

    def to(self, device):
        self.graph = self.graph.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.tree_partition = [partition.to(device) for partition in self.tree_partition]
        self.contrast_weight = [weight.to(device) for weight in self.contrast_weight]
        return self


class dblp_data():
    def __init__(self):
        self.x = None
        self.edge_index = None
        self.num_nodes = None
        self.y = None
        self.num_edges = None
        self.num_features = None


class dblp_dataset():
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes


def get_node_partition_index(hierarchical_partition):
    node_partition_index = []
    temp = None
    for k, partition in enumerate(hierarchical_partition):
        if k == 0:
            temp = partition
        else:
            temp = torch.sparse.mm(temp, partition)
        node_partition_index.append(temp._indices()[1])
    return node_partition_index


def get_con_weight(tree, t, big,dataname,pooling_type='entropy'):
    contrast_weights = []
    node_partition_index = get_node_partition_index(tree)
    if pooling_type == 'entropy':
        for k in range(0, len(tree)):
            if k == 0 and big:
                continue
            if k == len(tree) - 1:
                indices = torch.stack([torch.arange(len(node_partition_index[k - 1])), node_partition_index[k - 1]], dim=0)
                contrast_weight = torch.sparse_coo_tensor(indices, torch.zeros_like(node_partition_index[k - 1]),
                                                          size=(indices.shape[1], tree[k - 1].size(1)), dtype=torch.float32)
            else:
                contrast_weight = torch.t(tree[k].index_select(1, node_partition_index[k]))  # n*n. n*m
            contrast_weight = contrast_weight.coalesce()

            if k == 0:
                contrast_weight.values()[contrast_weight.indices()[0] == contrast_weight.indices()[1]] = 0
            else:
                diag_cols = node_partition_index[k - 1][contrast_weight.indices()[0]]
                update_mask = (contrast_weight.indices()[1] == diag_cols)
                contrast_weight.values()[update_mask] = 0
            unnormalized_weight = torch.pow(2, -1 * (contrast_weight.values() / t))
            contrast_weight = torch.sparse.FloatTensor(contrast_weight.indices(), unnormalized_weight,
                                                       contrast_weight.size())

            # norm
            contrast_weight = contrast_weight.coalesce()
            row_sum = contrast_weight.sum(dim=1).to_dense()
            normalized_weight = contrast_weight.values() / row_sum[contrast_weight.indices()[0]]
            # normalized_weight = torch.exp(contrast_weight.values()) / torch.exp(row_sum[contrast_weight.indices()[0]])
            contrast_weights.append(
                torch.sparse.FloatTensor(contrast_weight.indices(), normalized_weight, contrast_weight.size()))
        return contrast_weights

    elif pooling_type == "sum":
        for k in range(0, len(tree)):
            if k == 0 and big:
                continue
            if k == len(tree) - 1:
                indices = torch.stack([torch.arange(len(node_partition_index[k - 1])), node_partition_index[k - 1]], dim=0)
                contrast_weight = torch.sparse_coo_tensor(indices, torch.zeros_like(node_partition_index[k - 1]),
                                                          size=(indices.shape[1], tree[k - 1].size(1)), dtype=torch.float32)
            else:
                contrast_weight = torch.t(tree[k].index_select(1, node_partition_index[k]))  # n*n. n*m
            contrast_weight = contrast_weight.coalesce()
            unnormalized_weight = torch.ones_like(contrast_weight._values())
            contrast_weight = torch.sparse.FloatTensor(contrast_weight.indices(), unnormalized_weight,
                                                       contrast_weight.size())
            contrast_weights.append(contrast_weight)
        return contrast_weights
    elif pooling_type == "mean":
        for k in range(0, len(tree)):
            if k == 0 and big:
                continue
            if k == len(tree) - 1:
                indices = torch.stack([torch.arange(len(node_partition_index[k - 1])), node_partition_index[k - 1]],
                                      dim=0)
                contrast_weight = torch.sparse_coo_tensor(indices, torch.zeros_like(node_partition_index[k - 1]),
                                                          size=(indices.shape[1], tree[k - 1].size(1)),
                                                          dtype=torch.float32)
            else:
                contrast_weight = torch.t(tree[k].index_select(1, node_partition_index[k]))  # n*n. n*m

            contrast_weight = contrast_weight.coalesce()
            if dataname in ['Amazon-Clothing']:
                counts_tensor = torch.bincount(contrast_weight.indices()[1])
                row_means = 1 / counts_tensor
                unnormalized_weight = row_means[contrast_weight._indices()[1]]
            else:
                counts_tensor = torch.bincount(contrast_weight.indices()[0])
                row_means = 1 / counts_tensor
                unnormalized_weight = row_means[contrast_weight._indices()[0]]
            contrast_weight = torch.sparse.FloatTensor(contrast_weight.indices(), unnormalized_weight,
                                                       contrast_weight.size())
            contrast_weights.append(contrast_weight)
        return contrast_weights





def load_dataset(root=None, dataset_source=None):
    dataset = dblp_data()
    n1s = []
    n2s = []
    for line in open("{}/{}-network".format(root, dataset_source)):
        n1, n2 = line.strip().split('\t')
        if int(n1) > int(n2):
            n1s.append(int(n1))
            n2s.append(int(n2))

    num_nodes = max(max(n1s), max(n2s)) + 1
    print('nodes num', num_nodes)
    data_train = sio.loadmat("{}/{}-train".format(root, dataset_source))
    data_test = sio.loadmat("{}/{}-test".format(root, dataset_source))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    # dataset.edge_index=torch.tensor([n1s,n2s])
    dataset.edge_index = torch.tensor([n2s, n1s])
    dataset.y = labels
    dataset.x = features
    dataset.num_nodes = num_nodes
    dataset.num_edges = dataset.edge_index.shape[1]

    num_class = max(labels) + 1

    return dblp_dataset(dataset, num_classes=num_class)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return torch_sparse.SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sym_adj(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = sp.csr_matrix(([1]*edge_index.size(1), (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj



def load_data(dataset_name, tree_height, t, big):

    if dataset_name == 'CoraFull':
        dataset = CoraFull(root=data_pth + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=data_pth + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'dblp':
        dataset = load_dataset(root=data_pth + dataset_name ,dataset_source= dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Amazon-Clothing':
        dataset = load_dataset(root=data_pth + 'Clothing', dataset_source=dataset_name)
        num_nodes = dataset.data.num_nodes
    else:
        print("Dataset not support!")
        exit(0)

    data = dataset.data
    class_list = [i for i in range(dataset.num_classes)]
    print("********" * 10)

    train_num = class_split[dataset_name]["train"]
    val_num = class_split[dataset_name]["dev"]
    test_num = class_split[dataset_name]["test"]

    random.shuffle(class_list)
    train_class = class_list[: train_num]
    val_class = class_list[train_num: train_num + val_num]
    test_class = class_list[train_num + val_num:]
    print("train_num: {}; val_num: {}; test_num: {}".format(train_num, val_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)

    tree_path = data_pth + dataset_name + '/tree/' + str(tree_height) + '/'
    if (os.path.exists(tree_path + 'hierarchical_partition.pt')):
        tree_partition = torch.load(tree_path + 'hierarchical_partition.pt')
    else:
        tree_partition = get_tree_partition(dataset_name, data, tree_height, tree_path)  # 稀疏矩阵存编码树

    contrast_weight = get_con_weight(tree_partition, t, big,'entropy')

    # norm
    normalized_partition = []
    for partition in tree_partition:
        partition = partition.coalesce()
        row_sum = partition.sum(dim=0).to_dense()
        normalized_partition_value = partition.values() / row_sum[partition.indices()[1]]
        normalized_partition.append(
            torch.sparse.FloatTensor(partition.indices(), normalized_partition_value, partition.size()))

    edge_index_np = data.edge_index.numpy()
    coo_matrix = sp.coo_matrix((torch.ones(data.edge_index.size(1)), (edge_index_np[0], edge_index_np[1])),
                               shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
    adj2 = coo_matrix.tocsr()
    g = dgl.from_scipy(adj2)
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    if big:
        graph = sym_adj(data.edge_index, data.num_nodes)
        alpha = 0.05
        output = alpha * data.x
        for _ in range(5):
            data.x = torch.spmm(graph, data.x)
            output = output + data.x / 5
        data.x = output
    #################################
    # u, v = g.edges()
    # edge_index = torch.stack([u, v], dim=0).long()  # 确保是长整型
    # g = edge_index#Data(edge_index=edge_index)

    return DataSet(dataset=dataset_name, graph=g, x=data.x, y=data.y, num_class=dataset.num_classes,
                   tree_partition=normalized_partition, contrast_weight=contrast_weight, id_by_class=id_by_class,
                   train_class=train_class, val_class=val_class, test_class=test_class)


def tasks_generator(id_by_class, class_list, n_way, k_shot, m_query, num_task):
    x_spt = {}  # [0: array(1,2,5), 1: array(2,4,5)]
    x_qry = {}
    class_selected = {}  # {0: [1, 3, 5], 1: [2, 5, 6]}
    # sample class indices
    for i in range(num_task):
        class_selected[i] = random.sample(class_list, n_way)
        id_support = []
        id_query = []
        for cla in class_selected[i]:
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        x_spt[i] = np.array(id_support)
        x_qry[i] = np.array(id_query)

    return x_spt, x_qry, class_selected
