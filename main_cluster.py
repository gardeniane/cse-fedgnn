from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.optim as optim
import random
from models.gcn import GCN, GCN1
from models.gcn2 import GCN2, GCN3
import copy

import torch_geometric
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools
import torch_sparse

# %%
from data_process import generate_data, load_data, \
    load_data_new, partition_data_for_client, Data1
from train_func import test, test_cluster, test_hist, test_hist1, \
    train, train_cluster, train_histories_new, train_histories_new1, \
    Lhop_Block_matrix_train, FedSage_train, Communicate_train, train_histories
from torch_geometric.loader import ClusterData, ClusterLoader


import scipy.sparse as sp


def normalize(mx):  # adj matrix

    mx = mx + torch.eye(mx.shape[0], mx.shape[1])

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return torch.tensor(mx)

# 删除cross-cluster edges，包括一个client上的cross-cluster edges
def remove_intracluster_edge(edge_index, cluster_nodes):
    edge_index = edge_index.tolist()
    edge_num = len(edge_index[0])
    for i in range(edge_num-1,-1,-1):
        for cluster in cluster_nodes:
            if edge_index[0][i] not in cluster and edge_index[1][i] not in cluster:
                continue
            elif edge_index[0][i] in cluster and edge_index[1][i] in cluster:
                break
            else:
                edge_index[0].pop(i)
                edge_index[1].pop(i)

    return torch.tensor(edge_index)

# 只删除cross-client cross-cluster edges，保留within-client cross-cluster edges
def remove_crossclient_intracluster_edge(edge_index, cluster_nodes, client_node_index_list):
    edge_index = edge_index.tolist()
    edge_num = len(edge_index[0])
    for i in range(edge_num-1, -1, -1):
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


# 删除不参与本轮training的nodes，通过删除其对应的edges
def remove_not_training_nodes(edge_index, nodes_not_training):
    edge_index_copy = edge_index.clone()
    edge_index = edge_index.tolist()
    edge_num = len(edge_index[0])
    for i in range(edge_num-1,-1,-1):
        if edge_index[0][i] in nodes_not_training or edge_index[1][i] in nodes_not_training:
            edge_index[0].pop(i)
            edge_index[1].pop(i)

    return torch.tensor(edge_index)


# cross_client_neighbors: L-hop cross-client neighbors
def remove_intra_client_edge(client_adj_t, cross_client_neighbors_indexes):
    row, col, edge_attr = client_adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0).T.tolist()

    for i in range(len(edge_index)-1, -1, -1):
        if edge_index[i][0] in cross_client_neighbors_indexes \
                or edge_index[i][1] in cross_client_neighbors_indexes:
                edge_index.pop(i)

    return torch.tensor(edge_index).T

def remove_intra_client_edge_new(client_adj_t, cross_client_neighbors_indexes):
    row, col, edge_attr = client_adj_t.t().coo()


    edge_index = torch.stack([row, col], dim=0)
    num_nodes = int(edge_index.max()) + 1
    col, row = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    node_mask.fill_(True)
    node_mask[cross_client_neighbors_indexes] = False
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    return edge_index


# 找到两个tensor中只出现一次的元素
def setdiff1d(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # 计算过程应该是根据counts == 1得到index，然后再根据index取uniques的元素
    difference = uniques[counts == 1]
    return difference


# 找到两个tensor中都出现的元素
def intersect1d(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # 计算过程应该是根据counts > 1得到index，然后再根据index取uniques的元素
    intersection = uniques[counts > 1]
    return intersection


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

    neighbor_layer = copy.deepcopy(subsets)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    return neighbor_layer


def federated_GCN_embedding_update_periodic_cluster_update(data, client_node_index_list,
                                            cluster_partition, cluster_update_period,
                                            K, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period, participant_one_round):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN2(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()

    # Train model
    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_partition)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    adj = edgeindex_to_adj(data.edge_index.clone())

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes

    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                 L_hop, data.edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(), data.edge_index, L_hop)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = client_node_index_list[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], client_node_index_list[i])

            # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
            all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
            # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
            cross_client_neighbor_list.append(
                torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

        in_data_nei_indexes.append(all_cross_client_neighbor)
        in_com_train_nei_indexes.append(cross_client_neighbor_list)
        # client i上的邻接矩阵
        client_adj_t = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
        client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop-1])
        client_adj_t_partial_list.append(client_adj_t_partial)

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
    models = []
    optimizers = []
    for i in range(K):
        models.append(GCN2(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=communicate_indexes[i].shape[0]).to(device))

        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    # 将cluster的更新period相同的nodes合并到一起
    cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
    for i in range(len(cluster_partition)):
        cluster_nodes = cluster_partition[i]
        cluster_partition_group[cluster_update_period[i]] += cluster_nodes

    [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

    for t in range(global_epoch):
        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        for index in range(1, len(cluster_partition_group)):
            if cluster_partition_group[index] != []:
                if t >= 0 and (t + 1) % index == 0:
                    global_model.update_hists_cluster(cluster_partition_group[index])

        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print(client_participant_indexes)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for i in client_participant_indexes:
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                            features, adj, labels, communicate_indexes[i],
                                            in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                            in_com_train_nei_indexes[i], in_data_nei_indexes[i], client_node_index_list[i],
                                            in_com_train_local_node_indexes[i], global_model, period)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in client_participant_indexes:
            states.append(models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test],
                     'hop_federated_embedding_periodic_cluster_update_'+str(period)+'_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


def fgl_embedding_update_periodic_cluster_update(client_node_index_list,
                                            cluster_partition, cluster_update_period,
                                            K, data, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN3(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()
    models = []
    optimizers = []
    for i in range(K):
        models.append(GCN3(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=data.num_nodes).to(device))

        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    # 去掉cross-client between-cluster edges
    edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition, client_node_index_list)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    adj = edgeindex_to_adj(data.edge_index.clone())

    # Train model
    # 以下变量都是分clients保存的，注释默认指一个client
    communicate_indexes = []  # 保存local nodes+L-hop neighbors组成的subgraph
    in_com_train_data_indexes = []  # 保存train nodes在communicate_indexes中的indexes
    client_adj_t_partial_list = []  # 去掉cross-client neighbors的adj，用于第一层的emb.的计算
    in_data_nei_indexes = []  # 每层cross-client neighbors的实际indexes，也就是在data中的indexes
    in_com_train_nei_indexes = []  # 每层cross-client neighbors在communicate_indexes中的indexes
    in_com_train_local_node_indexes = []  # local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                 L_hop, data.edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors，indexes为0-L_hop，其中0index对应无元素
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(), data.edge_index, L_hop)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = client_node_index_list[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], client_node_index_list[i])

            # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
            all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
            # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
            cross_client_neighbor_list.append(
                torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

        in_data_nei_indexes.append(all_cross_client_neighbor)
        in_com_train_nei_indexes.append(cross_client_neighbor_list)
        # client i上的邻接矩阵
        client_adj_t = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
        client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop-1])
        client_adj_t_partial_list.append(client_adj_t_partial)

    # ###以下为local training

    for t in range(global_epoch):
        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        for j in range(len(cluster_update_period)):
            # 先根据cluster更新peridot更新cluster nodes的historical emb.
            if t > 0 and t % cluster_update_period[j] == 0:
                for i in range(K):
                    # 注意，只更新1-(L-1) hop的emb.
                    models[i].pull_latest_hists(global_model, in_data_nei_indexes[i][0])

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for i in range(K):
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                            features, adj, labels, communicate_indexes[i],
                                            in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                            in_com_train_nei_indexes[i], in_data_nei_indexes[i], client_node_index_list[i],
                                            in_com_train_local_node_indexes[i], global_model, period)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test],
                     'hop_federated_embedding_periodic_cluster_update_'+str(period)+'_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


def fgl_embedding_update_periodic_cluster_update_partial(client_node_index_list,
                                                 cluster_partition, cluster_update_period,
                                                 K, data, features, adj, labels,
                                                 idx_train, idx_val, idx_test,
                                                 iid_percent, L_hop, num_layers, period, participant_one_round):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN3(nfeat=features.shape[1],
                        nhid=args_hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args_dropout,
                        NumLayers=num_layers,
                        num_nodes=data.num_nodes).to(device)
    global_model.reset_parameters()
    models = []
    optimizers = []
    for i in range(K):
        models.append(GCN3(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=data.num_nodes).to(device))

        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    # 去掉cross-client between-cluster edges
    edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition, client_node_index_list)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    adj = edgeindex_to_adj(data.edge_index.clone())

    # Train model
    # 以下变量都是分clients保存的，注释默认指一个client
    communicate_indexes = []  # 保存local nodes+L-hop neighbors组成的subgraph
    in_com_train_data_indexes = []  # 保存train nodes在communicate_indexes中的indexes
    client_adj_t_partial_list = []  # 去掉cross-client neighbors的adj，用于第一层的emb.的计算
    in_data_nei_indexes = []  # 每层cross-client neighbors的实际indexes，也就是在data中的indexes
    in_com_train_nei_indexes = []  # 每层cross-client neighbors在communicate_indexes中的indexes
    in_com_train_local_node_indexes = []  # local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                                         L_hop, data.edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors，indexes为0-L_hop，其中0index对应无元素
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(), data.edge_index, L_hop)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = client_node_index_list[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], client_node_index_list[i])

            # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
            all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
            # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
            cross_client_neighbor_list.append(
                torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

        in_data_nei_indexes.append(all_cross_client_neighbor)
        in_com_train_nei_indexes.append(cross_client_neighbor_list)
        # client i上的邻接矩阵
        client_adj_t = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
        client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
        client_adj_t_partial_list.append(client_adj_t_partial)

    # ###以下为local training

    for t in range(global_epoch):
        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        for j in range(len(cluster_update_period)):
            # 先根据cluster更新peridot更新cluster nodes的historical emb.
            if t > 0 and t % cluster_update_period[j] == 0:
                for i in range(K):
                    # 注意，只更新1-(L-1) hop的emb.
                    models[i].pull_latest_hists(global_model, in_data_nei_indexes[i][0])

        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print(client_participant_indexes)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for i in client_participant_indexes:
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, communicate_indexes[i],
                                                 in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                 in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                 client_node_index_list[i],
                                                 in_com_train_local_node_indexes[i], global_model, period)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in client_participant_indexes:
            states.append(models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                     'hop_federated_embedding_periodic_cluster_update_' + str(period) + '_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

# 将每个epoch的结果，即loss+accuracy写到文件中
def write_result(epoch, iid_percent, L_hop, K, one_round_result, filename):
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = one_round_result
    a = open(dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + filename + str(
        num_layers) + 'layer_GCN_iter_' + str(global_epoch) + '_epoch_' + str(
        local_iteration) + '_device_num_' + str(K), 'a+')

    a.write(str(epoch) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
    a.write(str(epoch) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
    a.write(str(epoch) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
    a.close()
    # print("save file as", dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + 'hop_Block_federated_' + str(
    #     num_layers) + 'layer_GCN_iter_' + str(global_epoch) + '_epoch_' + str(local_iteration) + '_device_num_' + str(
    #     K))


def federated_GCN_cluster_partial(data, client_node_index_list, client_cluster_nodes, K, features, adj, labels,
                                  idx_train, idx_val, idx_test, iid_percent, L_hop,
                                  num_layers, participant_one_round):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers)
    global_model.reset_parameters()
    models = []
    for i in range(K):
        models.append(GCN(nfeat=features.shape[1],
                          nhid=args_hidden,
                          nclass=labels.max().item() + 1,
                          dropout=args_dropout,
                          NumLayers=num_layers))
    if args_cuda:
        for i in range(K):
            models[i] = models[i].cuda()
        global_model = global_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))
    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    cluster_num = 10
    cluster_data = ClusterData(data, num_parts=cluster_num)
    cluster_nodes = []  # 存放每个cluster的nodes
    for i in range(cluster_num):
        start = cluster_data.partptr[i]
        end = cluster_data.partptr[i + 1]
        cluster_nodes.append(cluster_data.perm[start:end].tolist())

    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_nodes)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]

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
    if args_cuda:
        iid_indexes = setdiff1d(torch.tensor(range(len(labels))).cuda(), torch.cat(split_data_indexes))
    else:
        iid_indexes = setdiff1d(torch.tensor(range(len(labels))), torch.cat(split_data_indexes))
    iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]

    for i in range(K):  # for iid
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) // average_device_of_class * (1 - non_iid_percent))
        split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes[:average_num])

        iid_indexes = iid_indexes[average_num:]
    # ###### 至此，为clients分配nodes完毕

    communicate_indexes = []
    in_com_train_data_indexes = []
    for i in range(K):
        communicate_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i], L_hop, data.edge_index)[0]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # only count the train data of nodes in current server(not communicate nodes)
        # inter = intersect1d(client_node_index_list[i], idx_train)  # 训练集idx_train中client i所包含的nodes集合
        #
        # in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
        #     torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    for t in range(global_epoch):
        client_cluster_selected = torch.ones(()).new_empty((cluster_num, K), dtype=torch.bool)
        client_cluster_selected.fill_(False)
        for c in range(cluster_num):
            client_participant = random.sample(range(0, K), participant_one_round)
            client_cluster_selected[c, :][client_participant] = True
        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        client_participant_indexes = []  # 用于记录本次参与training的clients
        in_com_train_data_indexes = []  # 为每个client记录参与training的nodes
        for i in range(K):
            # 为每个client选出参与本次training的nodes
            client_node_index_selected = torch.tensor([], dtype=int)
            for c in range(cluster_num):
                if client_cluster_selected[:, i][c]:
                    client_node_index_selected = torch.cat((client_node_index_selected, client_cluster_nodes[i][c]), 0)
            client_node_index_selected = client_node_index_selected.sort()[0]
            inter = intersect1d(client_node_index_selected, idx_train)  # 训练集idx_train中client i所包含的nodes集合

            in_com_train_data_indexes.append(torch.searchsorted(communicate_indexes[i], inter).clone())
            if len(in_com_train_data_indexes[i]) == 0:
                continue
            try:
                adj[communicate_indexes[i]][:, communicate_indexes[i]]
            except:  # adj is empty
                continue
            client_participant_indexes.append(i)
            for iteration in range(local_iteration):
                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                acc_train = Lhop_Block_matrix_train(iteration, models[i], optimizers[i],
                                                    features, adj, labels, communicate_indexes[i],
                                                    in_com_train_data_indexes[i])

            acc_trains.append(acc_train)  # 保存loss和accuracy

        print(client_participant_indexes)

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        # for i in range(K):
        for i in client_participant_indexes:
            states.append(models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                     'hop_federated_cluster_partition_partial_' + str(participant_one_round) + '_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

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

def federated_GCN_cluster_multimodel(data, cluster_partition, client_cluster_nodes, client_node_index_list,
                          K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_partition)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    adj = edgeindex_to_adj(data.edge_index.clone())

    cluster_num = 10
    global_model_list = []
    for j in range(cluster_num):
        global_model = GCN(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers)
        global_model.reset_parameters()
        global_model_list.append(global_model)

    local_models_list = []
    for i in range(K):
        models = []
        for j in range(cluster_num):
            models.append(GCN(nfeat=features.shape[1],
                              nhid=args_hidden,
                              nclass=labels.max().item() + 1,
                              dropout=args_dropout,
                              NumLayers=num_layers))
        local_models_list.append(models)

    # optimizer and train
    local_optimizers_list = []
    for i in range(K):
        optimizers = []
        for j in range(cluster_num):
            optimizers.append(optim.SGD(local_models_list[i][j].parameters(),
                                        lr=args_lr, weight_decay=args_weight_decay))
        local_optimizers_list.append(optimizers)

    # Train model
    communicate_indexes = []
    in_com_train_data_indexes = []
    for i in range(K):
        communicate_indexes_cluster = []
        in_com_train_data_indexes_cluster = []
        for j in range(cluster_num):
            communicate_index = torch_geometric.utils.k_hop_subgraph(client_cluster_nodes[i][j],
                                                                     L_hop, data.edge_index)[0]

            communicate_indexes_cluster.append(communicate_index)
            communicate_indexes_cluster[j] = communicate_indexes_cluster[j].sort()[0]
            # only count the train data of nodes in current server(not communicate nodes)
            inter = intersect1d(client_cluster_nodes[i][j], idx_train)  # 训练集idx_train中client i所包含的nodes集合
            in_com_train_data_indexes_cluster.append(
                torch.searchsorted(communicate_indexes_cluster[j], inter).clone())  # local id in block matrix

        communicate_indexes.append(communicate_indexes_cluster)
        in_com_train_data_indexes.append(in_com_train_data_indexes_cluster)

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        for j in range(cluster_num):
            local_models_list[i][j].load_state_dict(global_model_list[j].state_dict())

    for t in range(global_epoch):
        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for i in range(K):
            for j in range(cluster_num):
                for iteration in range(local_iteration):
                    if len(in_com_train_data_indexes[i][j]) == 0:
                        continue
                    try:
                        adj[communicate_indexes[i][j]][:, communicate_indexes[i][j]]
                    except:  # adj is empty
                        continue

                    # features, adj, labels等是整个dataset的数据
                    # 这里的communicate_indexes[i]是client i的training subgraph
                    # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                    acc_train = Lhop_Block_matrix_train(iteration, local_models_list[i][j], local_optimizers_list[i][j],
                                                        features, adj, labels, communicate_indexes[i][j],
                                                        in_com_train_data_indexes[i][j])

                # acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        for j in range(cluster_num):
            states = []  # 保存clients的local models
            gloabl_state = dict()
            for i in range(K):
                states.append(local_models_list[i][j].state_dict())
            # Average all parameters
            for key in global_model_list[j].state_dict():
                gloabl_state[key] = in_com_train_data_indexes[0][j].shape[0] * states[0][key]
                count_D = in_com_train_data_indexes[0][j].shape[0]
                for i in range(1, K):
                    gloabl_state[key] += in_com_train_data_indexes[i][j].shape[0] * states[i][key]
                    count_D += in_com_train_data_indexes[i][j].shape[0]
                gloabl_state[key] /= count_D

            global_model_list[j].load_state_dict(gloabl_state)  # 更新global model
            # ###至此global aggregation结束

        '''
        # 1.4 Testing
        avg_loss_train, avg_acc_train, avg_loss_val, avg_acc_val, avg_loss_test, avg_acc_test = 0, 0, 0, 0, 0, 0
        # print("total test nodes", idx_test.shape[0])
        for j in range(cluster_num):
            loss_train, acc_train = test(global_model_list[j], features, adj, labels,
                                         intersect1d(torch.tensor(cluster_partition[j]), idx_train))
            avg_loss_train += loss_train / cluster_num
            avg_acc_train += acc_train / cluster_num

            loss_val, acc_val = test(global_model_list[j], features, adj, labels,
                                     intersect1d(torch.tensor(cluster_partition[j]), idx_val))  # validation
            avg_loss_val += loss_val / cluster_num
            avg_acc_val += acc_val / cluster_num

            loss_test, acc_test = test(global_model_list[j], features, adj, labels,
                                       intersect1d(torch.tensor(cluster_partition[j]), idx_test))
            # print("test data nodes of cluster", "\t", intersect1d(torch.tensor(cluster_partition[j]), idx_test).shape[0])
            # print(t, '\t', "cluster", '\t', j, "test", '\t', loss_test, '\t', acc_test)
            avg_loss_test += loss_test / cluster_num
            avg_acc_test += acc_test / cluster_num
        

        print(t, '\t', "train", '\t', avg_loss_train, '\t', avg_acc_train)
        print(t, '\t', "val", '\t', avg_loss_val, '\t', avg_acc_val)
        print(t, '\t', "test", '\t', avg_loss_test, '\t', avg_acc_test)
        '''

        idx_train_cluster_list,idx_val_cluster_list, idx_test_cluster_list = [], [], []
        for j in range(cluster_num):
            idx_train_cluster_list.append(intersect1d(torch.tensor(cluster_partition[j]), idx_train))
            idx_val_cluster_list.append(intersect1d(torch.tensor(cluster_partition[j]), idx_val))
            idx_test_cluster_list.append(intersect1d(torch.tensor(cluster_partition[j]), idx_test))

        loss_train, acc_train = test_multi_model(global_model_list, features, adj, labels, idx_train, idx_train_cluster_list)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_multi_model(global_model_list, features, adj, labels, idx_val, idx_val_cluster_list)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_multi_model(global_model_list, features, adj, labels, idx_test, idx_test_cluster_list)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        # write_result(t, iid_percent, L_hop, K,
        #              [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test], 'hop_federated_cluster_')

        for i in range(K):
            for j in range(cluster_num):
                local_models_list[i][j].load_state_dict(global_model_list[j].state_dict())

    del features, adj, labels, idx_train, idx_val, idx_test
    while len(global_model_list) >= 1:
        del global_model_list[0]

    for i in range(K):
        while len(local_models_list[i]) >= 1:
            del local_models_list[i][0]


def federated_GCN_cluster(data, client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers)
    global_model.reset_parameters()

    edge_index_new = remove_intracluster_edge(data.edge_index, cluster_partition)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    adj = edgeindex_to_adj(data.edge_index.clone())

    models = []
    for i in range(K):
        models.append(GCN(nfeat=features.shape[1],
                          nhid=args_hidden,
                          nclass=labels.max().item() + 1,
                          dropout=args_dropout,
                          NumLayers=num_layers))
    if args_cuda:
        for i in range(K):
            models[i] = models[i].cuda()
        global_model = global_model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))
    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

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
    if args_cuda:
        iid_indexes = setdiff1d(torch.tensor(range(len(labels))).cuda(), torch.cat(split_data_indexes))
    else:
        iid_indexes = setdiff1d(torch.tensor(range(len(labels))), torch.cat(split_data_indexes))
    iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]

    for i in range(K):  # for iid
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) // average_device_of_class * (1 - non_iid_percent))
        split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes[:average_num])

        iid_indexes = iid_indexes[average_num:]
    # ###### 至此，为clients分配nodes完毕

    communicate_indexes = []
    in_com_train_data_indexes = []
    for i in range(K):
        communicate_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i], L_hop, edge_index)[0]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # only count the train data of nodes in current server(not communicate nodes)
        inter = intersect1d(client_node_index_list[i], idx_train)  # 训练集idx_train中client i所包含的nodes集合

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    for t in range(global_epoch):
        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for i in range(K):
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue

                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                acc_train = Lhop_Block_matrix_train(iteration, models[i], optimizers[i],
                                                    features, adj, labels, communicate_indexes[i],
                                                    in_com_train_data_indexes[i])

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test], 'hop_federated_cluster_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


def adj_dict2matrix(graph):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = torch.tensor(adj.toarray())
    adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
    return adj

# torch.manual_seed()为CPU设置随机数种子，torch.cuda.manual_seed()为GPU设置随机数种子
# random.seed()为random模块的随机数种子
# 作用类似于保持每次实验的现场，使得每次实验产生的随机数都相同，便于复现实验
# https://blog.csdn.net/weixin_44211968/article/details/123769010
# np.random.seed(42)
# torch.manual_seed(42)
# 'cora', 'citeseer', 'pubmed' #simulate #other dataset twitter,
dataset_name = "cora"  # 'ogbn-arxiv'

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

# %%
if dataset_name in ['simulate', 'cora', 'citeseer', 'pubmed']:
    args_hidden = 16
else:
    args_hidden = 256

args_dropout = 0.5
args_lr = 1.0
if dataset_name in ['cora', 'citeseer']:
    args_lr = 0.1
else:
    args_lr = 0.01

args_weight_decay = 5e-4  # L2 penalty
local_iteration = 3  # number of local training iterations
args_no_cuda = False
args_cuda = not args_no_cuda and torch.cuda.is_available()
num_layers = 2
interval_update = 40  # period更新周期

args_device_num = class_num  # split data into args_device_num parts
global_epoch = 200  # number of global rounds
device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
# %%
# for testing
# 先删除历史模型
path = "./"
model_files = os.listdir(path)
for i, f in enumerate(model_files):
    if f.find("cora_IID") >= 0:
        os.remove(path + f)

K = 20  # client_num
cluster_num = 5
client_node_index_list, cluster_partition, cluster_update_period, client_cluster_nodes = \
    partition_data_for_client(data, K, cluster_num, '')

# federated_GCN_cluster_partial(data, client_node_index_list, client_cluster_nodes, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers,
#                               participant_one_round=5)


cluster_update_period = np.ones(cluster_num)

# fgl_embedding_update_periodic_cluster_update(client_node_index_list, cluster_partition, cluster_update_period,
#                         K, data, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=1)

fgl_embedding_update_periodic_cluster_update_partial(client_node_index_list, cluster_partition, cluster_update_period,
                        K, data, features, adj, labels, idx_train, idx_val, idx_test, 1,
                         num_layers, num_layers, period=1, participant_one_round=5)

# federated_GCN_embedding_update_periodic_cluster_update(data, client_node_index_list, cluster_partition, cluster_update_period,
#                         K, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=40, participant_one_round=5)

# federated_GCN_embedding_update_periodic_cluster_update(client_node_index_list, cluster_partition, cluster_update_period,
#                         class_num, data, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=10)

# federated_GCN_cluster_multimodel(data, cluster_partition, client_cluster_nodes, client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          1, num_layers)

# federated_GCN_cluster(data, client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          1, num_layers)
