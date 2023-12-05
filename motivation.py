from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.optim as optim
import random
from models.gcn import GCN

import torch_geometric
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

import os
import copy

# %%
from data_process import generate_data, load_data, partition_data_for_client, load_data_new, Data1
from train_func import test, train, train_cluster
import scipy.sparse as sp
import networkx as nx
import itertools
import torch_sparse


# 找到两个tensor中都出现的元素
def intersect1d(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # 计算过程应该是根据counts > 1得到index，然后再根据index取uniques的元素
    intersection = uniques[counts > 1]
    return intersection


# 找到两个tensor中只出现一次的元素
def setdiff1d(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # 计算过程应该是根据counts == 1得到index，然后再根据index取uniques的元素
    difference = uniques[counts == 1]
    return difference


# 找到在t2但是不在t1中的元素
def t2_minus_t1(t1, t2):
    difference = set(np.array(t2)) - set(np.array(t1))
    return torch.tensor(list(difference))


# 只删除cross-client cross-cluster edges，保留within-client cross-cluster edges
def remove_crossclient_intracluster_edge(edge_index, cluster_nodes, client_node_index_list):
    edge_index = edge_index.tolist()
    edge_num = len(edge_index[0])
    for i in range(edge_num - 1, -1, -1):
        for cluster in cluster_nodes:
            # 其他clusters中的edges
            if edge_index[0][i] not in cluster and edge_index[1][i] not in cluster:
                continue
            # within-cluster edges
            elif edge_index[0][i] in cluster and edge_index[1][i] in cluster:
                break
            else:  # 一个node在当前cluster，一个在另外一个cluster
                within_client = False
                for client_node in client_node_index_list:  # 判断是否在一个client
                    if edge_index[0][i] in client_node and edge_index[1][i] in client_node:
                        within_client = True
                        break
                if within_client:
                    continue
                else:
                    edge_index[0].pop(i)
                    edge_index[1].pop(i)

    return torch.tensor(edge_index)


# 删除cross-cluster edges
def remove_intracluster_edge(edge_index, cluster_nodes):
    edge_index = copy.deepcopy(edge_index.tolist())
    edge_num = len(edge_index[0])
    count = 0
    for i in range(edge_num - 1, -1, -1):
        for cluster in cluster_nodes:
            # 其他clusters中的edges
            if edge_index[0][i] not in cluster and edge_index[1][i] not in cluster:
                continue
            # 当前cluster中的edges
            elif edge_index[0][i] in cluster and edge_index[1][i] in cluster:
                continue
            else:  # 一个node在当前cluster，一个在另外一个cluster
                edge_index[0].pop(i)
                edge_index[1].pop(i)
                count += 1

    print("count: ", count)

    return torch.tensor(edge_index)


def edgeindex_to_adj(edge_index):
    edge_list = edge_index.t().tolist()
    nodeset = sorted(set(itertools.chain(*edge_list)))
    g = nx.MultiGraph()  # 无向多边图，即一对nodes允许存在多条边
    g.add_nodes_from(nodeset)
    g.add_edges_from(edge_list)
    adj = sp.lil_matrix(nx.adjacency_matrix(g))
    adj = torch.tensor(adj.toarray())
    adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
    return adj


# 获取node_idx的num_hops neighbors，分layer保存
def get_all_Lhop_neighbors_new(node_idx, edge_index, num_hops):
    num_nodes = int(edge_index.max()) + 1
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    neighbor_layer = [node_idx]
    for hop in range(1, num_hops+1):
        neighbor_layer.append(intersect1d(torch.tensor([],dtype=int), subsets[hop]))

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    return neighbor_layer


# centralized_GCN + full batch
def centralized_GCN(features, adj, labels, idx_train, idx_val, idx_test, num_layers):
    model = GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers)
    model.reset_parameters()
    if args_cuda:
        # from torch_geometric.nn import DataParallel
        # model = DataParallel(model)
        # model= torch.nn.DataParallel(model)
        model = model.cuda()

        # features= torch.nn.DataParallel(features)

        features = features.cuda()

        # edge_index= torch.nn.DataParallel(edge_index)

        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    # optimizer and train

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args_lr, weight_decay=args_weight_decay)

    optimizer = optim.Adam(model.parameters(),
                         lr=args_lr, weight_decay=args_weight_decay)
    # Train model
    best_val = 0
    for t in range(global_epoch):  # make to equivalent to federated
        # for iteration in range(local_iteration):
        loss_train, acc_train = train(t, model, optimizer, features, adj, labels, idx_train)
        # validation
        loss_train, acc_train = test(model, features, adj, labels, idx_train)  # train after backward
        print(t, "train", loss_train, acc_train)
        loss_val, acc_val = test(model, features, adj, labels, idx_val)  # validation
        print(t, "val", loss_val, acc_val)
        # test
        loss_test, acc_test = test(model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        a = open(dataset_name + '_IID_' + 'centralized_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch),
                 'a+')
        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()


    print("save file as",
          dataset_name + '_IID_' + 'centralized_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch))

    del model, features, adj, labels, idx_train, idx_val, idx_test
    return loss_test, acc_test


def centralized_cluster_GCN(cluster_partition, data, features, adj, labels, idx_train, idx_val, idx_test, num_layers):
    model = GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers)
    model.reset_parameters()
    optimizer = optim.SGD(model.parameters(), lr=args_lr, weight_decay=args_weight_decay)
    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_partition)
    data.edge_index = None
    data.edge_index = edge_index_new
    adj = edgeindex_to_adj(data.edge_index.clone())
    for t in range(global_epoch):  # make to equivalent to federated
        # for iteration in range(local_iteration):
        loss_train, acc_train = train(t, model, optimizer, features, adj, labels, idx_train)
        # validation
        loss_train, acc_train = test(model, features, adj, labels, idx_train)  # train after backward
        print(t, "train", loss_train, acc_train)
        loss_val, acc_val = test(model, features, adj, labels, idx_val)  # validation
        print(t, "val", loss_val, acc_val)
        # test
        loss_test, acc_test = test(model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        a = open(dataset_name + '_IID_' + 'centralized_cluster_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch),
                 'a+')
        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()

    print("save file as",
          dataset_name + '_IID_' + 'centralized_cluster_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch))
    del model
    del features
    del adj
    del labels
    del idx_train
    del idx_val
    del idx_test

    return loss_test, acc_test


# 用于计算每种方式需要communication的node embeddings的数目
def federated_GCN_comm_nodes(K, adj, labels, iid_percent, L_hop, data, cluster_partition):
    # clustering之后的edge_index
    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_partition)
    adj_new = edgeindex_to_adj(data.edge_index.clone())

    row, col, edge_attr = adj.t().coo()
    edge_index = data.edge_index

    # ###### 以下为clients分配nodes
    # 1.1 按照labels将nodes分组，保存在shuffle_labels中
    nclass = labels.max().item() + 1
    split_data_indexes = []
    non_iid_percent = 1 - float(iid_percent)
    iid_indexes = []  # random assign
    shuffle_labels = []  # make train data points split into different devices
    for i in range(K):
        # torch.nonzero(labels == i) 取出labels中值为i的元素index，.reshape(-1)将tensor转成一行
        current = torch.nonzero(labels == i).reshape(-1)
        current = current[np.random.permutation(len(current))]  # shuffle
        shuffle_labels.append(current)

    # 1.2 本文每个client i对应一个主要label i。将该label对应的nodes分配部分给client，保存在split_data_indexes[i]中
    average_device_of_class = K // nclass
    if K % nclass != 0:  # for non-iid
        average_device_of_class += 1
    for i in range(K):
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) // average_device_of_class * non_iid_percent)
        split_data_indexes.append(
            (labels_class[average_num * (i % average_device_of_class):average_num * (i % average_device_of_class + 1)]))

    # 1.3 分配第二部分nodes给clients，最后更新split_data_indexes[i]
    iid_indexes = setdiff1d(torch.tensor(range(len(labels))), torch.cat(split_data_indexes))
    iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]

    for i in range(K):  # for iid
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) // average_device_of_class * (1 - non_iid_percent))
        split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes[:average_num])

        iid_indexes = iid_indexes[average_num:]
    # ###### 至此，为clients分配nodes完毕

    num_comm_emb, num_comm_emb_cluster = 0, 0
    for i in range(K):
        split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj

        # 返回split_data_indexes[i]+目标节点为split_data_indexes[i]的L_hop nodes组成的子图（即nodes set+edge set）
        # 这里[0]获取的是nodes集合
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                          data.edge_index, L_hop)

        neighbors_layer_cluster = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                     edge_index_new, L_hop)

        # for hop in range(1, L_hop+1):
        #     if neighbors_layer[hop].shape[0] < neighbors_layer_cluster[hop].shape[0]:
        #         print("")

        diff, diff_cluster = [], []
        index = 0
        for hop in range(2, L_hop+1):
            diff.append(t2_minus_t1(split_data_indexes[i], neighbors_layer[hop]))  # 需要communicated的nodes
            num_comm_emb += diff[index].shape[0]
            diff_cluster.append(t2_minus_t1(split_data_indexes[i], neighbors_layer_cluster[hop]))  # 需要communicated的nodes
            num_comm_emb_cluster += diff_cluster[index].shape[0]
            if num_comm_emb < num_comm_emb_cluster:
                print("error")

    print("num_comm_emb: ", num_comm_emb, "num_comm_emb_cluster: ",
          num_comm_emb_cluster, "comm_decay: ", (num_comm_emb - num_comm_emb_cluster) / num_comm_emb)

# 邻接表转邻接矩阵，稀疏矩阵
def adj_dict2matrix(graph):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = torch.tensor(adj.toarray())
    adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
    return adj

dataset_name = "ogbn-arxiv"  # 'ogbn-arxiv'

if dataset_name == 'simulate':
    number_of_nodes = 200
    class_num = 3
    link_inclass_prob = 10 / number_of_nodes  # when calculation , remove the link in itself
    # EGCN good when network is dense 20/number_of_nodes  #fails when network is sparse. 20/number_of_nodes/5

    link_outclass_prob = link_inclass_prob / 20

    features, adj, labels, idx_train, idx_val, idx_test = generate_data(number_of_nodes, class_num, link_inclass_prob,
                                                                        link_outclass_prob)
else:
    if dataset_name == "citeseer":
        features, adj, labels, idx_train, idx_val, idx_test, graph = load_data(dataset_name)
        data = Data1()
        edge_index = adj_dict2matrix(graph)
        row, col, edge_attr = edge_index.t().coo()
        edge_index = torch.stack([row, col], dim=0).T.tolist()
        data.edge_index = torch.tensor(edge_index).T
        # data_obj.edge_index = data_obj.edge_index.to(device)
        data.num_nodes = features.shape[0]
        data.num_edges = data.edge_index.shape[1]
    else:
        dataset, data, features, adj, labels, idx_train, idx_val, idx_test = load_data_new(dataset_name)
    class_num = labels.max().item() + 1  # 因为labels编号0-6
    # data.adj_t = gcn_norm(data.adj_t)

# %%
if dataset_name in ['simulate', 'cora', 'citeseer', 'pubmed']:
    args_hidden = 16
else:
    args_hidden = 256

args_dropout = 0.5
args_lr = 1.0
args_weight_decay = 5e-4  # L2 penalty
local_iteration = 3  # number of local training iterations
args_no_cuda = False
args_cuda = not args_no_cuda and torch.cuda.is_available()
num_layers = 2

args_device_num = class_num  # split data into args_device_num parts
global_epoch = 200  # number of global rounds
device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
# %%
# for testing
# 先删除历史模型
path = "./"
model_files = os.listdir(path)
for i, f in enumerate(model_files):
    if f.find(dataset_name + "_IID") >= 0:
        os.remove(path + f)

# if dataset_name in ['ogbn-arxiv']:
#     data.edge_index.device = device

# if dataset_name in ['cora', 'citeseer', 'pubmed']:
#     dataset = Planetoid(root="data/", name=dataset_name)
# else:
#     dataset = PygNodePropPredDataset(name='ogbn-arxiv')
# data = dataset[0]
K = 20  # client_num
cluster_num = 2
client_node_index_list, cluster_partition, cluster_update_period, \
            client_cluster_nodes = partition_data_for_client(data, K, cluster_num, '')

# 集中式GCN
# centralized_GCN(features, adj, labels, idx_train, idx_val, idx_test, num_layers)

# 集中式GCN + 去掉intra-cluster edges+部分客户端选择
# centralized_cluster_GCN(cluster_partition, data, features, adj, labels, idx_train, idx_val, idx_test, num_layers)

# for args_random_assign in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
federated_GCN_comm_nodes(K, adj, labels, 0.0, 5, data, cluster_partition)