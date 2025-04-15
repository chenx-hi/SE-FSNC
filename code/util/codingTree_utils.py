import copy
import json
import math
import heapq

import networkx as nx
import numba as nb
import numpy as np
import gc
import os
import scipy.sparse as sp
import torch
from scipy.sparse import load_npz, save_npz



def get_id():
    i = 0
    while True:
        yield i
        i += 1


@nb.jit(nopython=True)
def cut_volume(edge_set, p1, p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            if (p2[j] in edge_set[p1[i]]):
                c12 += edge_set[p1[i]][p2[j]]
    return c12


def merge(new_ID, id1, id2, cut_v, nowhigh, node_dict):
    n1 = node_dict[id1]
    n2 = node_dict[id2]
    new_partition = n1.partition + n2.partition
    v = n1.vol + n2.vol
    g = n1.g + n2.g - 2 * cut_v

    child = set()
    if (n1.children != None):
        child = child.union(n1.children)
    else:
        child.add(n1.ID)
    if (n2.children != None):
        child = child.union(n2.children)
    else:
        child.add(n2.ID)

    new_node = PartitionTreeNode(ID=new_ID, partition=new_partition, high=nowhigh, children=child, g=g, vol=v)

    if (n1.children != None):
        id1_child = n1.children
        for ID in id1_child:
            node_dict[ID].parent = new_ID
        del node_dict[id1]
    else:
        node_dict[id1].parent = new_ID
    if (n2.children != None):
        id2_child = n2.children
        for ID in id2_child:
            node_dict[ID].parent = new_ID
        del node_dict[id2]
    else:
        node_dict[id2].parent = new_ID

    node_dict[new_ID] = new_node


def graph_parse(adj_matrix):
    row = adj_matrix.row
    col = adj_matrix.col
    weight = adj_matrix.data
    g_num_nodes = adj_matrix.shape[0]
    VOL = np.sum(weight)
    node_vol = np.zeros(g_num_nodes)
    Int = nb.types.int32
    Float = nb.types.float64
    ValueDict = nb.types.DictType(Int, Float)
    edge_set = nb.typed.typeddict.Dict.empty(Int, ValueDict)
    adj_table = {}
    edgeNum = 0
    for i in range(g_num_nodes):
        adj = set()
        adj_table[i] = adj
        edge_set[i] = nb.typed.typeddict.Dict.empty(Int, Float)
    for i in range(len(row)):
        if (not col[i] in edge_set[row[i]]):
            edge_set[row[i]][col[i]] = weight[i]
            adj_table[row[i]].add(col[i])
            node_vol[row[i]] += weight[i]
            edgeNum += 1
        if (not row[i] in edge_set[col[i]]):
            edge_set[col[i]][row[i]] = weight[i]
            adj_table[col[i]].add(row[i])
            node_vol[col[i]] += weight[i]
            edgeNum += 1

        # if(not col[i] in edge_set[row[i]]):
        #     edge_set[row[i]][col[i]] = weight[i]
        #     adj_table[row[i]].add(col[i])
        #     node_vol[row[i]] += weight[i]
        #     edgeNum += 1

    print('edge_number：' + str(edgeNum))
    return g_num_nodes, VOL, node_vol, adj_table, edge_set, edgeNum


class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, high=1, children: set = None, parent=None, entropy=0.0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.high = high
        self.entropy = entropy

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())


class PartitionTree():
    def __init__(self, adj_matrix):
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table, self.edge_set, self.edgeNum = graph_parse(
            adj_matrix)  # adj_table
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()
        self.PartitionTreeEntropy = 0

    def build_leaves(self):
        node_vol = self.node_vol
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g=v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def __build_k_tree(self, g_vol, nodes_dict: dict, k=2):
        nowhigh = 2

        while (nowhigh <= k):
            if (nowhigh == 2):
                nodes_ids = nodes_dict.keys()
                edge_set = self.edge_set
                adj_table = self.adj_table
            else:
                old_new_dict = {}
                new_old_dict = {}
                nodes_ids = []
                nodes_ids_for_return = []
                new_nodes_dict = {}
                Int = nb.types.int32
                Float = nb.types.float64
                ValueDict = nb.types.DictType(Int, Float)
                new_edge_set = nb.typed.typeddict.Dict.empty(Int, ValueDict)
                new_edge_set_for_return = nb.typed.typeddict.Dict.empty(Int, ValueDict)
                adj_table = {}
                for key, value in nodes_dict.items():
                    if (value.parent == None):
                        newID = next(self.id_g)
                        new_leaf_node = PartitionTreeNode(ID=newID, partition=[newID], g=value.g, vol=value.vol,
                                                          high=nowhigh - 1)
                        old_new_dict[key] = newID
                        new_old_dict[newID] = key
                        nodes_ids.append(newID)
                        nodes_ids_for_return.append(key)
                        new_nodes_dict[newID] = new_leaf_node
                nodes_dict.update(new_nodes_dict)
                for i in range(len(nodes_ids)):
                    adj = set()
                    adj_table[nodes_ids[i]] = adj
                    new_edge_set[nodes_ids[i]] = nb.typed.typeddict.Dict.empty(Int, Float)
                    new_edge_set_for_return[nodes_ids_for_return[i]] = nb.typed.typeddict.Dict.empty(Int, Float)
                startIDList = list(edge_set.keys())
                for startID in startIDList:
                    if (nodes_dict[startID].parent != None):
                        startParent_for_return = nodes_dict[startID].parent
                    else:
                        startParent_for_return = startID
                    startParent = old_new_dict[startParent_for_return]
                    weightDict = edge_set[startID]
                    endIDList = list(weightDict.keys())
                    for endID in endIDList:
                        if (nodes_dict[endID].parent != None):
                            endParent_for_return = nodes_dict[endID].parent
                        else:
                            endParent_for_return = endID
                        endParent = old_new_dict[endParent_for_return]
                        weight = weightDict[endID]
                        if (not endParent in adj_table[startParent]):
                            adj_table[startParent].add(endParent)
                        if (endParent in new_edge_set[startParent]):
                            new_edge_set[startParent][endParent] += weight
                            new_edge_set_for_return[startParent_for_return][endParent_for_return] += weight
                        else:
                            new_edge_set[startParent][endParent] = weight
                            new_edge_set_for_return[startParent_for_return][endParent_for_return] = weight

                edge_set = new_edge_set

            min_heap = []
            new_id = None
            node1List = []
            node2List = []
            cutvList = []
            v1List = []
            v2List = []
            g1List = []
            g2List = []
            for i in nodes_ids:
                for j in adj_table[i]:
                    if j > i:
                        node1List.append(i)
                        node2List.append(j)
                        n1 = nodes_dict[i]
                        n2 = nodes_dict[j]
                        v1List.append(n1.vol + 1)
                        v2List.append(n2.vol + 1)
                        g1List.append(n1.g + 1)
                        g2List.append(n2.g + 1)
                        if len(n1.partition) == 1 and len(n2.partition) == 1:
                            cut_v = 0
                            if (n2.partition[0] in edge_set[n1.partition[0]]):
                                cut_v += edge_set[n1.partition[0]][n2.partition[0]]
                        else:
                            cut_v = cut_volume(edge_set, p1=np.array(n1.partition), p2=np.array(n2.partition))
                        cutvList.append(cut_v)
            v1 = np.array(v1List)
            v2 = np.array(v2List)
            g1 = np.array(g1List)
            g2 = np.array(g2List)
            cutvList = np.array(cutvList)
            v12 = v1 + v2
            g12 = g1 + g2 - 2 * cutvList
            diffList = ((v12 - g12) * np.log2(v12) - (v1 - g1) * np.log2(v1) - (v2 - g2) * np.log2(v2) + (
                        g12 - g1 - g2) * np.log2(g_vol)) / g_vol
            min_heap = []
            for i in range(len(diffList)):
                if (diffList[i] <= 0):
                    heapq.heappush(min_heap, (diffList[i], node1List[i], node2List[i], cutvList[i]))

            merged_count = 0
            while merged_count > -1:
                if len(min_heap) == 0:
                    break
                diff, id1, id2, cut_v = heapq.heappop(min_heap)

                if not id1 in nodes_dict or not id2 in nodes_dict:
                    continue
                if nodes_dict[id1].merged or nodes_dict[id2].merged:
                    continue
                if nodes_dict[id1].g == 0.0 or nodes_dict[id2].g == 0.0:
                    continue
                nodes_dict[id1].merged = True
                nodes_dict[id2].merged = True
                new_id = next(self.id_g)
                merge(new_id, id1, id2, cut_v, nowhigh, nodes_dict)
                adj_table[new_id] = adj_table[id1].union(adj_table[id2])
                for node in nodes_dict[new_id].partition:
                    if node in adj_table[new_id]:
                        adj_table[new_id].remove(node)
                del adj_table[id1]
                del adj_table[id2]

                merged_count += 1
                if (merged_count % 50000 == 0):
                    print('gc working')
                    gc.collect()

                nodeList = []
                new_cutvList = []
                vList = []
                gList = []
                n2 = nodes_dict[new_id]
                n2partition = np.array(n2.partition)
                n2vol = n2.vol + 1
                n2g = n2.g + 1
                for ID in adj_table[new_id]:
                    if not nodes_dict[ID].merged:
                        nodeList.append(ID)
                        n1 = nodes_dict[ID]
                        vList.append(n1.vol + 1)
                        gList.append(n1.g + 1)
                        cut_v = cut_volume(edge_set, np.array(n1.partition), n2partition)
                        new_cutvList.append(cut_v)
                v1 = np.array(vList)
                v2 = np.int32(n2vol)
                g1 = np.array(gList)
                g2 = np.int32(n2g)
                new_cutvList = np.array(new_cutvList)
                v12 = v1 + v2
                g12 = g1 + g2 - 2 * new_cutvList
                new_diffList = ((v12 - g12) * np.log2(v12) - (v1 - g1) * np.log2(v1) - (v2 - g2) * np.log2(v2) + (
                            g12 - g1 - g2) * np.log2(g_vol)) / g_vol
                for i in range(len(new_diffList)):
                    if (new_diffList[i] <= 0):
                        heapq.heappush(min_heap, (new_diffList[i], nodeList[i], new_id, new_cutvList[i]))

            if (nowhigh == 2):
                new_nodes_dict = {}
                for key, value in nodes_dict.items():
                    if (value.parent == None and value.high == 1):
                        newID = next(self.id_g)
                        children = set()
                        children.add(key)
                        new_leaf_node = PartitionTreeNode(ID=newID, partition=value.partition, g=value.g, vol=value.vol,
                                                          high=nowhigh, parent=None, children=children)
                        value.parent = newID
                        new_nodes_dict[newID] = new_leaf_node
                nodes_dict.update(new_nodes_dict)

            if (nowhigh > 2):
                for key, value in nodes_dict.items():
                    if (nodes_dict[key].high == nowhigh):
                        old_child_set = nodes_dict[key].children
                        new_children = set()
                        partition = []
                        for child in old_child_set:
                            new_children.add(new_old_dict[child])
                            partition = partition + nodes_dict[new_old_dict[child]].partition

                        nodes_dict[key].children = new_children
                        nodes_dict[key].partition = partition

                for key, value in old_new_dict.items():
                    if (nodes_dict[value].parent == None):
                        # print('AAAAAAAAAA')
                        nodes_dict[value].high = nowhigh
                        children = set()
                        children.add(key)
                        nodes_dict[value].children = children
                        nodes_dict[value].partition = nodes_dict[key].partition
                        nodes_dict[key].parent = value


                    else:
                        nodes_dict[key].parent = nodes_dict[value].parent
                        del nodes_dict[value]

                edge_set = new_edge_set_for_return

            nowhigh += 1

        rootID = next(self.id_g)
        rootchild = set()
        for key, value in nodes_dict.items():
            if (value.parent != None):
                continue
            else:
                rootchild.add(key)
                value.parent = rootID
        rootNode = PartitionTreeNode(ID=rootID, partition=[], vol=self.VOL, g=0, children=rootchild, high=nowhigh)
        nodes_dict[rootID] = rootNode
        return rootID

    def build_coding_tree(self, k):
        if k == 1:
            print('Error treehigh')
            return
        else:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k)

    def create_node_entropy(self):
        rootID = self.root_id
        IDlist = []
        glist = []
        vlist = []
        v_father_list = []
        VOL = self.VOL
        for k, v in self.tree_node.items():
            if (v.ID == rootID):
                continue
            IDlist.append(v.ID)
            glist.append(v.g)
            vlist.append(v.vol)
            v_father_list.append(self.tree_node[v.parent].vol)
        glist = np.array(glist)
        vlist = np.array(vlist)
        v_father_list = np.array(v_father_list)
        entropyList = -(glist / VOL) * np.log2(vlist / v_father_list)
        for i in range(len(IDlist)):
            if (np.isnan(entropyList[i])):
                self.tree_node[IDlist[i]].entropy = 0.00001
            elif (entropyList[i] == 0.0 or entropyList[i] == -0.0):
                self.tree_node[IDlist[i]].entropy = 0.00001
            else:
                self.tree_node[IDlist[i]].entropy = entropyList[i]


# save node
def printPartitionTree(partitionTree, path):
    with open(path, 'w') as f:
        for k, v in partitionTree.tree_node.items():
            f.write(str(v) + '\n')
    f.close()


# encoding tree by matrix
def partitionTreeFromMatrix(matrix, treehigh):
    partitiontree = PartitionTree(adj_matrix=matrix)
    partitiontree.build_coding_tree(treehigh)
    # partitiontree.build_coding_tree(4)
    partitiontree.create_node_entropy()
    return partitiontree


# build encoding tree if not
def build_k_coding_tree(dataset, matrix, treehigh, tree_path):
    path = tree_path + dataset + '_' + str(treehigh) + '.txt'

    if os.path.exists(path):
        f = open(path)
        nodesData = f.readlines()
        tree_node = {}
        for i in range(len(nodesData)):
            if (nodesData[i] == ''):
                break
            nodedata = nodesData[i]
            ID = int(nodedata.split('ID=')[1].split(',partition')[0])
            partitionNodes = nodedata.split('[')[1].split(']')[0]
            if (partitionNodes == ''):
                partition = []
            else:
                partitionNodes = partitionNodes.split(', ')
                partition = [int(j) for j in partitionNodes]
            parent = nodedata.split('parent=')[1].split(',children')[0]
            if (parent == 'None'):
                parent = None
            else:
                parent = int(parent)
            childrenNodes = nodedata.split('children=')[1].split(',vol')[0]
            if (childrenNodes == 'None'):
                children = None
            else:
                childrenNodes = childrenNodes.split('{')[1].split('}')[0].split(', ')
                children = set(int(j) for j in childrenNodes)
            vol = float(nodedata.split('vol=')[1].split(',g')[0])
            g = float(nodedata.split('g=')[1].split(',merged')[0])
            high = int(nodedata.split('high=')[1].split(',entropy')[0])
            entropy = float(nodedata.split('entropy=')[1].split('}')[0])
            node = PartitionTreeNode(ID, partition, vol, g, high, children, parent, entropy)
            tree_node[ID] = node

        tree = PartitionTree(matrix)
        tree.tree_node = tree_node

    else:
        tree = partitionTreeFromMatrix(matrix, treehigh)
        printPartitionTree(tree, path)

    return tree

def get_children(nodes_dict, node_id):
    return list(nodes_dict[node_id].children)


def get_hierarchical_partition(n_nodes, nodes_dict, treehigh, tree_path):
    parent_list = []
    for i in range(n_nodes):
        current_node = nodes_dict[i]
        node_parent_list = []
        for j in range(treehigh):
            node_parent_list.append(current_node.parent)
            current_node = nodes_dict[current_node.parent]
        parent_list.append(node_parent_list)
    parent_list = np.array(parent_list)
    pnodes_id = [sorted(set(parent_list[:, i])) for i in range(treehigh)]
    hierarchical_partition = []
    last_high_num = n_nodes
    for k in range(treehigh):
        indices = [[],[]]
        values = []
        if k == 0:
            for i, parent_id in enumerate(pnodes_id[k]):
                children = get_children(nodes_dict, parent_id)
                for j, child_id in enumerate(children):
                    indices[0].append(child_id)
                    indices[1].append(i)
                    values.append(nodes_dict[child_id].entropy)
            sorted_indices_0, sorting_order = torch.sort(torch.tensor(indices[0]))
            sorted_indices_1 = torch.index_select(torch.tensor(indices[1]), dim=0, index=sorting_order)
            sorted_values = torch.index_select(torch.tensor(values), dim=0, index=sorting_order)

            thisHigh_w = torch.sparse_coo_tensor(torch.tensor([sorted_indices_0.tolist(),sorted_indices_1.tolist()]),
                                                 sorted_values, size=(last_high_num, len(pnodes_id[k])))

            last_high_num = len(pnodes_id[k])
        else:
            for i, parent_id in enumerate(pnodes_id[k]):
                children = get_children(nodes_dict, parent_id)
                for j, child_id in enumerate(children):
                    children_loc = pnodes_id[k - 1].index(child_id)
                    indices[0].append(children_loc)
                    indices[1].append(i)
                    values.append(nodes_dict[child_id].entropy)
            sorted_indices_0, sorting_order = torch.sort(torch.tensor(indices[0]))
            sorted_indices_1 = torch.index_select(torch.tensor(indices[1]), dim=0, index=sorting_order)
            sorted_values = torch.index_select(torch.tensor(values), dim=0, index=sorting_order)
            thisHigh_w = torch.sparse_coo_tensor(torch.tensor([sorted_indices_0.tolist(),sorted_indices_1.tolist()]),
                                                 sorted_values, size=(last_high_num, len(pnodes_id[k])))

            last_high_num = len(pnodes_id[k])
        hierarchical_partition.append(thisHigh_w.to(torch.float32))
    torch.save(hierarchical_partition, tree_path + 'hierarchical_partition.pt')
    return hierarchical_partition


def get_tree_partition(dataset, data, tree_height, tree_path):
    os.makedirs(os.path.dirname(tree_path), exist_ok=True)
    edgeWeight = np.ones(len(data.edge_index[0]))
    # adj = sp.coo_matrix((edgeWeight, (data.edge_index[0], data.edge_index[1])),
    #                        shape=[data.num_nodes, data.num_nodes])
    adj = sp.coo_matrix((edgeWeight, (data.edge_index[0], data.edge_index[1])),
                        shape=[data.x.shape[0], data.x.shape[0]])
    encoding_tree = build_k_coding_tree(dataset, np.round(adj, decimals=2), tree_height, tree_path)
    tree_partition = get_hierarchical_partition(encoding_tree.g_num_nodes, encoding_tree.tree_node, tree_height, tree_path)
    return tree_partition

