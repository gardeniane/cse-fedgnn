from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.optim as optim
import random
from models.gcn import GCN, GCN1
from models.gcn2 import GCN2, GCN3, GCN4
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
import datetime

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

# 找到在t2但是不在t1中的元素
def t2_minus_t1(t1, t2):
    difference = set(np.array(t2)) - set(np.array(t1))
    return torch.tensor(list(difference))


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
    for hop in range(1, num_hops + 1):
        neighbor_layer.append(intersect1d(torch.tensor([], dtype=int), subsets[hop]))

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    return neighbor_layer

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


def adj_dict2matrix(graph):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = torch.tensor(adj.toarray())
    adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
    return adj


def random_fl(data, client_node_index_list, K, features, adj, labels,
                                  idx_train, idx_val, idx_test, iid_percent, L_hop,
                                  num_layers, participant_one_round):
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

    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    # 去除cross-client cross-cluster edges
    # edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition,
    #                                                            client_node_index_list)
    # dataset.data.edge_index = None
    # dataset.data.edge_index = edge_index_new
    # data = dataset[0]
    # adj = edgeindex_to_adj(data.edge_index)

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                                         L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                     communicate_edge_index, L_hop)
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

    models = []
    for i in range(K):
        models.append(GCN3(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=data.num_nodes).to(device))

    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    num_comm_emb_list = []  # 记录每个clients的cross-client neighbors
    # 1个float 4字节，1个node emb.的维度为args_hidden=16，带宽1GBps，
    # 则传输1个node emb的时间为8*16/(1024*1024) s
    time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
    for i in range(K):
        num_comm_emb = 0
        local_nodes = torch.tensor(client_node_index_list[i])
        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        local_nodes = local_nodes.sort()[0]

        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                          data.edge_index, L_hop)

        diff, diff_cluster = [], []
        index = 0
        for hop in range(1, L_hop):
            diff.append(t2_minus_t1(local_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
            num_comm_emb += diff[index].shape[0]

        num_comm_emb_list.append(num_comm_emb)

        time_comm_list.append(num_comm_emb*4*16*1000*1000/(1024*1024))

    total_comm_nodes = 0  # 记录通信的nodes的个数
    epoch_runtime = []  # 记录每个epoch的runtime
    computation_time = []  # 记录每个epoch的计算时间
    communnication_time = []  # 记录每个epoch的通信时间
    for t in range(global_epoch):

        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print("selected clients:", client_participant_indexes)

        for i in client_participant_indexes:
            # 注意，只更新1-(L-1) hop的emb.
            models[i].pull_latest_hists(global_model, in_data_nei_indexes[i][0])

        # 更新selected clients的historical emb.
        for index in client_participant_indexes:
            total_comm_nodes += num_comm_emb_list[index]
            global_model.update_hists_cluster(client_node_index_list[index])

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        # for i in range(K):
        max_runtime_epoch = 0
        max_comp_epoch = 0
        max_comm_epoch = 0
        for i in client_participant_indexes:
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue

                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                start_time = datetime.datetime.now()
                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, communicate_indexes[i],
                                                 in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                 in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                 client_node_index_list[i],
                                                 in_com_train_local_node_indexes[i], global_model, 1)
                end_time = datetime.datetime.now()
                update_delay = (end_time - start_time).microseconds
                if time_comm_list[i] >= max_comm_epoch:
                    max_comm_epoch = time_comm_list[i]
                if update_delay >= max_comp_epoch:
                    max_comp_epoch = update_delay

        computation_time.append(max_comp_epoch)
        communnication_time.append(max_comm_epoch)
        epoch_runtime.append(max_comp_epoch + max_comm_epoch)

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
        loss_train, acc_train = test_hist1(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                     'random_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    total_comp_time = np.array(computation_time).sum()
    tot_comm_time = np.array(communnication_time).sum()
    total_runtime = np.array(epoch_runtime).sum()

    print("total_comp_time:",total_comp_time,"tot_comm_time:",tot_comm_time,"total_runtime:",total_runtime)

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


def digest(data, client_node_index_list, K, features, adj, labels,
                                  idx_train, idx_val, idx_test, iid_percent, L_hop,
                                  num_layers, period, participant_one_round):
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

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    # 去除cross-client cross-cluster edges
    # edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition,
    #                                                            client_node_index_list)
    # dataset.data.edge_index = None
    # dataset.data.edge_index = edge_index_new
    # data = dataset[0]
    # adj = edgeindex_to_adj(data.edge_index)

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                                         L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                     communicate_edge_index, L_hop)
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

    models = []
    for i in range(K):
        models.append(GCN2(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=communicate_indexes[i].shape[0]).to(device))

    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    num_comm_emb_list = []  # 记录每个clients的cross-client neighbors
    # 1个float 4字节，1个node emb.的维度为args_hidden=16，带宽1GBps，
    # 则传输1个node emb的时间为8*16/(1024*1024) s
    time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
    for i in range(K):
        num_comm_emb = 0
        local_nodes = torch.tensor(client_node_index_list[i])
        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        local_nodes = local_nodes.sort()[0]

        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                          data.edge_index, L_hop)

        diff, diff_cluster = [], []
        index = 0
        for hop in range(1, L_hop):
            diff.append(t2_minus_t1(local_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
            num_comm_emb += diff[index].shape[0]

        num_comm_emb_list.append(num_comm_emb)

        time_comm_list.append(num_comm_emb*4*16*1000*1000/(1024*1024))

    total_comm_nodes = 0  # 记录通信的nodes的个数
    epoch_runtime = []  # 记录每个epoch的runtime
    computation_time = []  # 记录每个epoch的计算时间
    communnication_time = []  # 记录每个epoch的通信时间
    for t in range(global_epoch):
        if t >= 0 and (t + 1) % period == 0:  # 更新所有nodes的historical emb.
            global_model.update_hists_cluster(torch.tensor(range(features.shape[0])))
            communnication_time.append(np.array(time_comm_list).max())

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        max_runtime_epoch = 0
        max_comp_epoch = 0
        max_comm_epoch = 0
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print("selected clients:", client_participant_indexes)
        # for i in range(K):
        for i in client_participant_indexes:
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue

                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                start_time = datetime.datetime.now()
                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, communicate_indexes[i],
                                                 in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                 in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                 client_node_index_list[i],
                                                 in_com_train_local_node_indexes[i], global_model, 1)
                end_time = datetime.datetime.now()
                update_delay = (end_time - start_time).microseconds
                # if time_comm_list[i] >= max_comm_epoch:
                #     max_comm_epoch = time_comm_list[i]
                if update_delay >= max_comp_epoch:
                    max_comp_epoch = update_delay

        computation_time.append(max_comp_epoch)
        epoch_runtime.append(max_comp_epoch)

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
            # for i in range(1, K):
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
                     'digest_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    total_comp_time = np.array(computation_time).sum()
    tot_comm_time = np.array(communnication_time).sum()
    total_runtime = np.array(epoch_runtime).sum()

    print("total_comp_time:",total_comp_time,"tot_comm_time:",tot_comm_time,"total_runtime:",total_runtime)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def adaptive(data, client_node_index_list, K, features, adj, labels,
                                  idx_train, idx_val, idx_test, iid_percent, L_hop,
                                  num_layers, participant_one_round, period_list):
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

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    # 去除cross-client cross-cluster edges
    # edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition,
    #                                                            client_node_index_list)
    # dataset.data.edge_index = None
    # dataset.data.edge_index = edge_index_new
    # data = dataset[0]
    # adj = edgeindex_to_adj(data.edge_index)

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                                         L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                     communicate_edge_index, L_hop)
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

    models = []
    for i in range(K):
        models.append(GCN2(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=communicate_indexes[i].shape[0]).to(device))

    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    num_comm_emb_list = []  # 记录每个clients的cross-client neighbors
    # 1个float 4字节，1个node emb.的维度为args_hidden=16，带宽1GBps，
    # 则传输1个node emb的时间为8*16/(1024*1024) s
    time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
    for i in range(K):
        num_comm_emb = 0
        local_nodes = torch.tensor(client_node_index_list[i])
        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        local_nodes = local_nodes.sort()[0]

        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                          data.edge_index, L_hop)

        diff, diff_cluster = [], []
        index = 0
        for hop in range(1, L_hop):
            diff.append(t2_minus_t1(local_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
            num_comm_emb += diff[index].shape[0]

        num_comm_emb_list.append(num_comm_emb)

        time_comm_list.append(num_comm_emb*4*16*1000*1000/(1024*1024))

    total_comm_nodes = 0  # 记录通信的nodes的个数
    epoch_runtime = []  # 记录每个epoch的runtime
    computation_time = []  # 记录每个epoch的计算时间
    communnication_time = []  # 记录每个epoch的通信时间
    stage_len = (global_epoch // len(period_list))
    for t in range(global_epoch):
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print("selected clients:", client_participant_indexes)

        stage = t // stage_len
        if (t >= 0 and (t + 1 - stage * stage_len) % period_list[stage]) or \
                (t + 1 - stage * stage_len == stage_len) == 0:  # 更新所有nodes的historical emb.
            global_model.update_hists_cluster(torch.tensor(range(features.shape[0])))
            communnication_time.append(np.array(time_comm_list).max())

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        max_runtime_epoch = 0
        max_comp_epoch = 0
        max_comm_epoch = 0
        # for i in range(K):
        for i in client_participant_indexes:
            for iteration in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue

                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue

                start_time = datetime.datetime.now()
                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, communicate_indexes[i],
                                                 in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                 in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                 client_node_index_list[i],
                                                 in_com_train_local_node_indexes[i], global_model, 1)
                end_time = datetime.datetime.now()
                update_delay = (end_time - start_time).microseconds
                # if time_comm_list[i] >= max_comm_epoch:
                #     max_comm_epoch = time_comm_list[i]
                if update_delay >= max_comp_epoch:
                    max_comp_epoch = update_delay

        computation_time.append(max_comp_epoch)
        epoch_runtime.append(max_comp_epoch)

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
            # for i in range(1, K):
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
                     'adaptive_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    total_comp_time = np.array(computation_time).sum()
    tot_comm_time = np.array(communnication_time).sum()
    total_runtime = np.array(epoch_runtime).sum()

    print("total_comp_time:",total_comp_time,"tot_comm_time:",tot_comm_time,"total_runtime:",total_runtime)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def Lhop_Block_federated_GCN(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
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
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj

        # 返回split_data_indexes[i]+目标节点为split_data_indexes[i]的L_hop nodes组成的子图（即nodes set+edge set）
        # 这里[0]获取的是nodes集合
        communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i], L_hop, edge_index)[0]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # only count the train data of nodes in current server(not communicate nodes)
        inter = intersect1d(split_data_indexes[i], idx_train)  # 训练集idx_train中client i所包含的nodes集合

        # torch.searchsorted(sortedsequence=[1,3,5,7,9], values=[3,6,9])返回一个和values一样大小的tensor,
        # 其中的元素是在sorted_sequence中大于等于values中值的索引
        # 本例中输出为[1,3,4]，其中1，4分别是value=3，9的index，而3则是(7>=6)得到的
        # 因为communicate_indexes[i]包含split_data_indexes[i]，而split_data_indexes[i]包含inter，
        # 所以这里返回的是精准查找的结果，即communicate_indexes[i]中出现inter的index
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

        a = open(dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + 'hop_Block_federated_' + str(
            num_layers) + 'layer_GCN_iter_' + str(local_iteration) + '_epoch_' + str(
            global_epoch) + '_device_num_' + str(K), 'a+')

        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

# 就是client内部仍然是2-layer，cross-client只选1-hop neighbor
def BDS_GCN(data, client_node_index_list, K, features, adj, labels,
                      idx_train, idx_val, idx_test, iid_percent,
                      sample_rate=0.5, L_hop=1,
                      num_layers=2):
    # K: number of models
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN4(nfeat=features.shape[1],
                        nhid=args_hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args_dropout,
                        NumLayers=num_layers,
                        num_nodes=data.num_nodes).to(device)
    global_model.reset_parameters()

    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    # 去除cross-client cross-cluster edges
    # edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition,
    #                                                            client_node_index_list)
    # dataset.data.edge_index = None
    # dataset.data.edge_index = edge_index_new
    # data = dataset[0]
    # adj = edgeindex_to_adj(data.edge_index)

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(client_node_index_list[i],
                                                                                         L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(client_node_index_list[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], client_node_index_list[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                     communicate_edge_index, L_hop)
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
        # 用于计算(num_layers-1)-hop上的nodes的emb.，即去掉num_layers-hop的cross-client neighbors相关edges
        if L_hop == num_layers:
            client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][num_layers - 1])
        else:
            client_adj_t_partial = copy.deepcopy(client_adj_t)
        client_adj_t_partial_list.append(client_adj_t_partial)

    models = []
    for i in range(K):
        models.append(GCN4(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=communicate_indexes[i].shape[0]).to(device))

    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    num_comm_emb_list = []  # 记录每个clients的cross-client neighbors
    # 1个float 4字节，1个node emb.的维度为args_hidden=16，带宽1GBps，
    # 则传输1个node emb的时间为8*16/(1024*1024) s
    time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
    diff, diff_cluster = [], []
    for i in range(K):
        num_comm_emb = 0
        local_nodes = torch.tensor(client_node_index_list[i])
        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        local_nodes = local_nodes.sort()[0]

        diff.append(t2_minus_t1(local_nodes, communicate_indexes[i]))  # 需要communicated的nodes
        num_comm_emb += diff[i].shape[0]

        num_comm_emb_list.append(num_comm_emb)

        # 为啥乘了2个1000？我忘记了
        time_comm_list.append(num_comm_emb * 4 * 16 * 1000 * 1000 / (1024 * 1024))

    total_comm_nodes = 0  # 记录通信的nodes的个数
    epoch_runtime = []  # 记录每个epoch的runtime
    computation_time = []  # 记录每个epoch的计算时间
    communnication_time = []  # 记录每个epoch的通信时间
    for t in range(global_epoch):

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        max_runtime_epoch = 0
        max_comp_epoch = 0
        max_comm_epoch = 0
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print("selected clients:", client_participant_indexes)
        # for i in range(K):
        for i in client_participant_indexes:
            if len(in_com_train_data_indexes[i]) == 0:
                continue
            try:
                adj[communicate_indexes[i]][:, communicate_indexes[i]]
            except:  # adj is empty
                continue

            # diff[i]上保存需要从其他clients上采样的nodes
            # torch.randperm(n)：将0至n-1（包括0和n - 1）随机打乱后获得的数字序列
            # sample_index是local nodes（split_data_indexes[i]）+diff的部分
            random_sample = diff[i][torch.randperm(len(diff[i]))[:int(len(diff[i]) * sample_rate)]]
            sample_index = torch.cat(
                (client_node_index_list[i], random_sample)).clone()

            sample_index = sample_index.sort()[0]
            # client i上的local training nodes
            inter = intersect1d(client_node_index_list[i],
                                idx_train)  ###only count the train data of nodes in current server(not communicate nodes)
            in_sample_train_data_index = torch.searchsorted(sample_index, inter).clone()  # local id in block matrix
            # ### 至此结束，其他也都一样，只不过把in_com_train_data_indexes替换成了in_sample_train_data_index
            if len(in_sample_train_data_index) == 0:
                continue
            try:
                adj[sample_index][:, sample_index]
            except:  # adj is empty
                continue

            # in_com_train_data_indexes保存无需通信的nodes在communicate_index的index集合
            in_com_train_data_index = torch.searchsorted(sample_index, inter).clone()  # local id in block matrix
            client_adj_t_partial = adj[sample_index][:, sample_index]
            in_com_train_local_node = torch.searchsorted(sample_index, client_node_index_list[i]).clone()
            # 分层保存所有neighbors
            neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(),
                                                         communicate_edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor = torch.searchsorted(sample_index, torch.tensor(random_sample).sort()[0]).clone()

            communnication_time.append(np.array(time_comm_list[i]).max())
            for iteration in range(local_iteration):

                start_time = datetime.datetime.now()
                #
                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, sample_index,
                                                 in_com_train_data_index, client_adj_t_partial,
                                                 cross_client_neighbor, random_sample,
                                                 client_node_index_list[i],
                                                 in_com_train_local_node, global_model, 1)
                end_time = datetime.datetime.now()
                update_delay = (end_time - start_time).microseconds
                # if time_comm_list[i] >= max_comm_epoch:
                #     max_comm_epoch = time_comm_list[i]
                if update_delay >= max_comp_epoch:
                    max_comp_epoch = update_delay

            # client训练完之后更新自己client上的node embeddings
            global_model.update_hists_cluster(client_node_index_list[i])

        computation_time.append(max_comp_epoch)
        epoch_runtime.append(max_comp_epoch)

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
            # for i in range(1, K):
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
                     'BDS_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    total_comp_time = np.array(computation_time).sum()
    tot_comm_time = np.array(communnication_time).sum()
    total_runtime = np.array(epoch_runtime).sum()

    print("total_comp_time:", total_comp_time, "tot_comm_time:", tot_comm_time, "total_runtime:", total_runtime)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def BDS_GCN2(data, client_node_index_list, K, features, adj, labels,
                      idx_train, idx_val, idx_test, iid_percent,
                      sample_rate=0.5, L_hop=1,
                      num_layers=2):
    # K: number of models
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN4(nfeat=features.shape[1],
                        nhid=args_hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args_dropout,
                        NumLayers=num_layers,
                        num_nodes=data.num_nodes).to(device)
    global_model.reset_parameters()

    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    # 去除cross-client cross-cluster edges
    # edge_index_new = remove_crossclient_intracluster_edge(data.edge_index, cluster_partition,
    #                                                            client_node_index_list)
    # dataset.data.edge_index = None
    # dataset.data.edge_index = edge_index_new
    # data = dataset[0]
    # adj = edgeindex_to_adj(data.edge_index)

    nclass = labels.max().item() + 1
    split_data_indexes = []
    non_iid_percent = 1 - float(iid_percent)
    iid_indexes = []  # random assign
    shuffle_labels = []  # make train data points split into different devices
    for i in range(K):
        current = torch.nonzero(labels == i).reshape(-1)
        current = current[np.random.permutation(len(current))]  # shuffle
        shuffle_labels.append(current)

    average_device_of_class = K // nclass
    if K % nclass != 0:  # for non-iid
        average_device_of_class += 1
    for i in range(K):
        label_i = i // average_device_of_class
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) // average_device_of_class * non_iid_percent)
        split_data_indexes.append(
            (labels_class[average_num * (i % average_device_of_class):average_num * (i % average_device_of_class + 1)]))

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

    communicate_indexes = []
    in_com_train_data_indexes = []
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):

        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
                                                                                         L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(split_data_indexes[i], idx_train)

        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], split_data_indexes[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors_new(split_data_indexes[i].clone(),
                                                     communicate_edge_index, L_hop)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = split_data_indexes[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], split_data_indexes[i])

            # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
            all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
            # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
            cross_client_neighbor_list.append(
                torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

        in_data_nei_indexes.append(all_cross_client_neighbor)
        in_com_train_nei_indexes.append(cross_client_neighbor_list)
        # client i上的邻接矩阵
        client_adj_t = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 用于计算(num_layers-1)-hop上的nodes的emb.，即去掉num_layers-hop的cross-client neighbors相关edges
        if L_hop == num_layers:
            client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][num_layers - 1])
        else:
            client_adj_t_partial = copy.deepcopy(client_adj_t)
        client_adj_t_partial_list.append(client_adj_t_partial)

    models = []
    for i in range(K):
        models.append(GCN4(nfeat=features.shape[1],
                           nhid=args_hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args_dropout,
                           NumLayers=num_layers,
                           num_nodes=communicate_indexes[i].shape[0]).to(device))

    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    num_comm_emb_list = []  # 记录每个clients的cross-client neighbors
    # 1个float 4字节，1个node emb.的维度为args_hidden=16，带宽1GBps，
    # 则传输1个node emb的时间为8*16/(1024*1024) s
    time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
    diff, diff_cluster = [], []
    for i in range(K):
        num_comm_emb = 0
        local_nodes = torch.tensor(split_data_indexes[i])
        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        local_nodes = local_nodes.sort()[0]

        diff.append(t2_minus_t1(local_nodes, communicate_indexes[i]))  # 需要communicated的nodes
        num_comm_emb += diff[i].shape[0]

        num_comm_emb_list.append(num_comm_emb)

        # 为啥乘了2个1000？我忘记了
        time_comm_list.append(num_comm_emb * 4 * 16 * 1000 * 1000 / (1024 * 1024))

    total_comm_nodes = 0  # 记录通信的nodes的个数
    epoch_runtime = []  # 记录每个epoch的runtime
    computation_time = []  # 记录每个epoch的计算时间
    communnication_time = []  # 记录每个epoch的通信时间
    for t in range(global_epoch):

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        max_runtime_epoch = 0
        max_comp_epoch = 0
        max_comm_epoch = 0
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print("selected clients:", client_participant_indexes)
        # for i in range(K):
        for i in client_participant_indexes:
            if len(in_com_train_data_indexes[i]) == 0:
                continue
            try:
                adj[communicate_indexes[i]][:, communicate_indexes[i]]
            except:  # adj is empty
                continue

            # diff[i]上保存需要从其他clients上采样的nodes
            # torch.randperm(n)：将0至n-1（包括0和n - 1）随机打乱后获得的数字序列
            # sample_index是local nodes（split_data_indexes[i]）+diff的部分
            random_sample = diff[i][torch.randperm(len(diff[i]))[:int(len(diff[i]) * sample_rate)]]
            sample_index = torch.cat(
                (split_data_indexes[i], random_sample)).clone()

            sample_index = sample_index.sort()[0]
            # client i上的local training nodes
            inter = intersect1d(split_data_indexes[i],
                                idx_train)  ###only count the train data of nodes in current server(not communicate nodes)
            in_sample_train_data_index = torch.searchsorted(sample_index, inter).clone()  # local id in block matrix
            # ### 至此结束，其他也都一样，只不过把in_com_train_data_indexes替换成了in_sample_train_data_index
            if len(in_sample_train_data_index) == 0:
                continue
            try:
                adj[sample_index][:, sample_index]
            except:  # adj is empty
                continue

            # in_com_train_data_indexes保存无需通信的nodes在communicate_index的index集合
            in_com_train_data_index = torch.searchsorted(sample_index, inter).clone()  # local id in block matrix
            client_adj_t_partial = adj[sample_index][:, sample_index]
            in_com_train_local_node = torch.searchsorted(sample_index, split_data_indexes[i]).clone()
            # 分层保存所有neighbors
            neighbors_layer = get_all_Lhop_neighbors_new(split_data_indexes[i].clone(),
                                                         communicate_edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor = torch.searchsorted(sample_index, torch.tensor(random_sample).sort()[0]).clone()

            communnication_time.append(np.array(time_comm_list[i]).max())
            for iteration in range(local_iteration):

                start_time = datetime.datetime.now()
                #
                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                                 features, adj, labels, sample_index,
                                                 in_com_train_data_index, client_adj_t_partial,
                                                 cross_client_neighbor, random_sample,
                                                 split_data_indexes[i],
                                                 in_com_train_local_node, global_model, 1)
                end_time = datetime.datetime.now()
                update_delay = (end_time - start_time).microseconds
                # if time_comm_list[i] >= max_comm_epoch:
                #     max_comm_epoch = time_comm_list[i]
                if update_delay >= max_comp_epoch:
                    max_comp_epoch = update_delay

            # client训练完之后更新自己client上的node embeddings
            global_model.update_hists_cluster(split_data_indexes[i])

        computation_time.append(max_comp_epoch)
        epoch_runtime.append(max_comp_epoch)

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
            # for i in range(1, K):
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
                     'BDS_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    total_comp_time = np.array(computation_time).sum()
    tot_comm_time = np.array(communnication_time).sum()
    total_runtime = np.array(epoch_runtime).sum()

    print("total_comp_time:", total_comp_time, "tot_comm_time:", tot_comm_time, "total_runtime:", total_runtime)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


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
args_lr = 0.1
# if dataset_name in ['cora', 'citeseer']:
#     args_lr = 0.1
# else:
#     args_lr = 0.1

args_weight_decay = 5e-4  # L2 penalty
local_iteration = 3  # number of local training iterations
args_no_cuda = False
args_cuda = not args_no_cuda and torch.cuda.is_available()
num_layers = 2
interval_update = 40  # period更新周期

args_device_num = class_num  # split data into args_device_num parts
global_epoch = 500  # number of global rounds
device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
# %%
# for testing
# 先删除历史模型
path = "./"
model_files = os.listdir(path)
for i, f in enumerate(model_files):
    if f.find(dataset_name + "_IID") >= 0:
        os.remove(path + f)

K = 20  # client_num
cluster_num = 5
client_node_index_list, cluster_partition, cluster_update_period, client_cluster_nodes = \
    partition_data_for_client(data, K, cluster_num, '')
#
# data = Planetoid(root="data/", name=dataset_name)[0]

# federated_GCN_cluster_partial(data, client_node_index_list, client_cluster_nodes, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers,
#                               participant_one_round=5)


participant_one_round = 5  # K for pubmed

# random_fl(data, client_node_index_list, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 2, num_layers,
#                               participant_one_round)

# adaptive(data, client_node_index_list, K, features, adj, labels,
#          idx_train, idx_val, idx_test, 1, 2, num_layers,
#          participant_one_round, [10, 10, 15, 20, 30])  # 200 epoches分成4个stages，前面的interval小，后面的大

# digest(data, client_node_index_list, K, features, adj, labels,
#                                   idx_train, idx_val, idx_test, 0, 2,
#                                   num_layers, 40, participant_one_round)


BDS_GCN(data, client_node_index_list, K, features, adj, labels,
                                idx_train, idx_val, idx_test, 0)

# BDS_GCN2(data, client_node_index_list, K, features, adj, labels,
#                                 idx_train, idx_val, idx_test, 0)


# Lhop_Block_federated_GCN(K, features, adj, labels, idx_train, idx_val, idx_test, 0,
#                          0, num_layers=2)