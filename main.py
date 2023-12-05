from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from models.gcn import GCN, GCN1
from models.gcn2 import GCN2
import copy

import torch_geometric
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

import os

# %%
from data_process import generate_data, load_data, partition_data_for_client
from train_func import test, test_hist, test_hist1, \
    train, train_cluster, train_histories_new, train_histories_new1, \
    Lhop_Block_matrix_train, FedSage_train, train_histories
from torch_geometric.loader import ClusterData
from torch_geometric.data import DataLoader


# %%
def get_K_hop_neighbors(adj_matrix, index, K):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0], adj_matrix.shape[1])  # make sure the diagonal part >= 1
    hop_neightbor_index = index
    for i in range(K):
        hop_neightbor_index = torch.unique(torch.nonzero(adj[hop_neightbor_index])[:, 1])
    return hop_neightbor_index


import scipy.sparse as sp


def normalize(mx):  # adj matrix

    mx = mx + torch.eye(mx.shape[0], mx.shape[1])

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return torch.tensor(mx)


# %% md
# Model
# %%
# define model

# for compare 2-10 layer performance in appendix
# iterations = 400
# Adam, lr = 0.01


def federated_cluster_gcn(data):
    cluster_data = ClusterData(data, num_parts=4)



def centralized_cluster_GCN(data, features, adj, labels, idx_train, idx_val, idx_test, num_layers):
    model = GCN(nfeat=features.shape[1],
                nhid=args_hidden,
                nclass=labels.max().item() + 1,
                dropout=args_dropout,
                NumLayers=num_layers)
    model.reset_parameters()
    optimizer = optim.SGD(model.parameters(), lr=args_lr, weight_decay=args_weight_decay)
    cluster_data = ClusterData(data, num_parts=local_iteration)
    for t in range(global_epoch):  # make to equivalent to federated
        # for iteration in range(local_iteration):
        batch_index = random.randint(0, local_iteration-1)
        split_data_indexes = list(range(cluster_data.partptr[batch_index],
                                      cluster_data.partptr[batch_index + 1]))
        split_data_index_list = cluster_data.perm[split_data_indexes]
        # split_data_index_list = split_data_indexes
        # random.shuffle(split_data_index_list)
        # split_data_index_list = torch.tensor(split_data_index_list)
        print('batch_index:', batch_index, 'split_data_index_list:',
              len(split_data_index_list))
        communicate_index = torch_geometric.utils.k_hop_subgraph(
            split_data_index_list, num_layers, data.edge_index)[0]
        communicate_index = communicate_index.sort()[0]
        inter = intersect1d(split_data_index_list, idx_train)
        in_com_train_data_index = torch.searchsorted(communicate_index, inter).clone()
        if len(in_com_train_data_index) == 0:
            continue
        # train
        print('train:')
        loss_train, acc_train = train_cluster(
            t, model, optimizer, features, adj, labels, idx_train, in_com_train_data_index, communicate_index)

        # validation
        loss_train, acc_train = test(model, features, adj, labels, idx_train)  # train after backward
        print(t, "train", loss_train, acc_train)
        loss_val, acc_val = test(model, features, adj, labels, idx_val)  # validation
        print(t, "val", loss_val, acc_val)
        # loss_test, acc_test = test_cluster(model, data, labels)
        # print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        a = open(dataset_name + '_IID_' + 'centralized_cluster_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch),
                 'a+')
        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        # a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()

    print("save file as",
          dataset_name + '_IID_' + 'cluster_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch))
    del model
    del features
    del adj
    del labels
    del idx_train
    del idx_val
    del idx_test

    return loss_test, acc_test


def train_mini(model, optimizer, train_loader):
    model.train()
    loss_all = 0
    len_train_dataset = 0
    for sub_data in train_loader:  # Iterate over each mini-batch.
        out = model(sub_data.x, sub_data.edge_index)  # Perform a single forward pass.
        # Compute the loss solely based on the training nodes.
        loss = F.nll_loss(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
        loss.backward()  # Derive gradients.
        loss_all += loss.item()
        len_train_dataset += len(sub_data.train_mask)
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    return (loss_all / len_train_dataset)


def train2(model, optimizer, data):
    model.train()
    loss_all = 0
    len_train_dataset = 0
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    return loss


def test_mini(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
    return accs


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

    # adj_t = client_adj_t.clone()
    # for node_index in cross_client_neighbors_indexes.tolist():
    #     adj_t[node_index, :].zero_()  # 将index为node_index的行置为0
    #     adj_t[:, node_index].zero_()  # 将index为node_index的列置为0
    # return adj_t


# centralized_GCN + mini-batch
def centralized_cluster_GCN_mini(dataset, data, features, adj, labels, idx_train, idx_val, idx_test, num_layers):
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

    optimizer = optim.SGD(model.parameters(),
                          lr=args_lr, weight_decay=args_weight_decay)

    cluster_num = 10
    cluster_data = ClusterData(data, num_parts=cluster_num)
    cluster_nodes = []  # 存放每个cluster的nodes
    for i in range(cluster_num):
        start = cluster_data.partptr[i]
        end = cluster_data.partptr[i + 1]
        cluster_nodes.append(cluster_data.perm[start:end].tolist())

    edge_index_new = remove_intracluster_edge(data, cluster_nodes)
    dataset.data.edge_index = None
    dataset.data.edge_index = edge_index_new
    data = dataset[0]
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train model
    best_val = 0
    for t in range(global_epoch):  # make to equivalent to federated
        # for iteration in range(local_iteration):
        loss_train = train2(model, optimizer, data)

        # validation
        loss_train, acc_train = test(model, features, adj, labels, idx_train)  # train after backward
        print(t, "train", loss_train, acc_train)
        loss_val, acc_val = test(model, features, adj, labels, idx_val)  # validation
        print(t, "val", loss_val, acc_val)
        # test
        loss_test, acc_test = test(model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        a = open(dataset_name + '_IID_' + 'centralized_cluster_mini_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch),
                 'a+')
        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()


    print("save file as",
          dataset_name + '_IID_' + 'centralized_cluster_mini_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch))
    del model
    del features
    del adj
    del labels
    del idx_train
    del idx_val
    del idx_test

    return loss_test, acc_test

# centralized_GCN + mini-batch
def centralized_GCN_mini(dataset, data, features, adj, labels, idx_train, idx_val, idx_test, num_layers):
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

    optimizer = optim.SGD(model.parameters(),
                          lr=args_lr, weight_decay=args_weight_decay)

    # 不适用于cora等这种只有1个图的数据集，这里的batch_size是子图个数
    # 因此无论设置batch_size为多少，最终train_loader都是一整个数据集
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Train model
    best_val = 0
    for t in range(global_epoch):  # make to equivalent to federated
        # for iteration in range(local_iteration):
        loss_train = train_mini(model, optimizer, train_loader)

        # validation
        loss_train, acc_train = test(model, features, adj, labels, idx_train)  # train after backward
        print(t, "train", loss_train, acc_train)
        loss_val, acc_val = test(model, features, adj, labels, idx_val)  # validation
        print(t, "val", loss_val, acc_val)
        # test
        loss_test, acc_test = test(model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        a = open(dataset_name + '_IID_' + 'centralized_mini_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch),
                 'a+')
        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()

    print("save file as",
          dataset_name + '_IID_' + 'centralized_mini_' + str(num_layers) + 'layer_GCN_iter_' + str(global_epoch))
    del model
    del features
    del adj
    del labels
    del idx_train
    del idx_val
    del idx_test

    return loss_test, acc_test

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

    optimizer = optim.SGD(model.parameters(),
                          lr=args_lr, weight_decay=args_weight_decay)

    # optimizer = optim.Adam(model.parameters(),
    #                      lr=args_lr, weight_decay=args_weight_decay)
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


# %distributed GCN with neighboring communication
def BDS_federated_GCN(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, sample_rate=0.5, L_hop=1,
                      num_layers=2):
    # K: number of models
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

    split_data_indexes = []

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

    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj

        communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i], L_hop, edge_index)[0]
        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        inter = intersect1d(split_data_indexes[i],
                            idx_train)  ###only count the train data of nodes in current server(not communicate nodes)
        in_com_train_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

    # assign global model weights to local models at initial step
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    for t in range(global_epoch):
        acc_trains = []
        for i in range(K):
            for epoch in range(local_iteration):
                # ### 注意：这是跟其他算法不一样的地方：部分采样cross-client neighbors，其他的全都一样
                # 需要从其他clients上采样的nodes
                diff = setdiff1d(split_data_indexes[i], communicate_indexes[i])
                # torch.randperm(n)：将0至n-1（包括0和n - 1）随机打乱后获得的数字序列
                # sample_index是local nodes（split_data_indexes[i]）+diff的部分
                sample_index = torch.cat(
                    (split_data_indexes[i], diff[torch.randperm(len(diff))[:int(len(diff) * sample_rate)]])).clone()

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

                acc_train = FedSage_train(i, epoch, models[i], optimizers[i],
                                          features, adj, labels, sample_index, in_sample_train_data_index)
            acc_trains.append(acc_train)

        states = []
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

        global_model.load_state_dict(gloabl_state)

        # Testing

        loss_train, acc_train = test(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test], 'BDS_federated_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


# %%
def FedSage_plus(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop=1, num_layers=2):
    # K: number of models
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args_cuda:
        for i in range(K):
            models[i] = models[i].to(device)
        global_model = global_model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    # optimizer and train
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(),
                                    lr=args_lr, weight_decay=args_weight_decay))
    # Train model

    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)

    split_data_indexes = []

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
        iid_indexes = setdiff1d(torch.tensor(range(len(labels))).to(device), torch.cat(split_data_indexes))
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
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).to(device)
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj

        communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i], L_hop, edge_index)[0]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        inter = intersect1d(split_data_indexes[i],
                            idx_train)  ###only count the train data of nodes in current server(not communicate nodes)

        in_com_train_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

    features_in_clients = []
    # assume the linear generator learnt the optimal (the average of features of neighbor nodes)
    # gaussian noise
    for i in range(K):
        # orignial features of outside neighbors of nodes in client i
        original_feature_i = features[setdiff1d(split_data_indexes[i], communicate_indexes[i])].clone()

        gaussian_feature_i = original_feature_i + torch.normal(0, 0.1, original_feature_i.shape).to(device)

        copy_feature = features.clone()

        copy_feature[setdiff1d(split_data_indexes[i], communicate_indexes[i])] = gaussian_feature_i

        features_in_clients.append(copy_feature[communicate_indexes[i]])
        print(features_in_clients[i].shape, communicate_indexes[i].shape)

    # assign global model weights to local models at initial step
    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    for t in range(global_epoch):
        acc_trains = []
        for i in range(K):
            for epoch in range(local_iteration):
                if len(in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    adj[communicate_indexes[i]][:, communicate_indexes[i]]
                except:  # adj is empty
                    continue
                acc_train = FedSage_train(epoch, models[i], optimizers[i],
                                          features_in_clients[i], adj, labels, communicate_indexes[i],
                                          in_com_train_data_indexes[i])

            acc_trains.append(acc_train)
        states = []
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

        global_model.load_state_dict(gloabl_state)

        # Testing

        loss_train, acc_train = test(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        a = open(dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + 'hop_FedSage_' + str(
            num_layers) + 'layer_GCN_iter_' + str(global_epoch) + '_epoch_' + str(
            local_iteration) + '_device_num_' + str(K), 'a+')

        a.write(str(t) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(t) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.close()
        for i in range(K):
            models[i].load_state_dict(gloabl_state)
    # test
    loss_test, acc_test = test(global_model, features, adj, labels, idx_test)
    print(t, '\t', "test", '\t', loss_test, '\t', acc_test)
    a = open(dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + 'hop_FedSage_' + str(
        num_layers) + 'layer_GCN_iter_' + str(global_epoch) + '_epoch_' + str(local_iteration) + '_device_num_' + str(K),
             'a+')
    a.write(str(t) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
    a.close()
    print("save file as", dataset_name + '_IID_' + str(iid_percent) + '_' + str(L_hop) + 'hop_FedSage_' + str(
        num_layers) + 'layer_GCN_iter_' + str(global_epoch) + '_epoch_' + str(local_iteration) + '_device_num_' + str(K))

    del global_model
    del features
    del adj
    del labels
    del idx_train
    del idx_val
    del idx_test
    while (len(models) >= 1):
        del models[0]

    return loss_test, acc_test


# L_hop_nodes: split_data_indexes[i],第L-hop nodes
def get_all_Lhop_neighbors(L_hop_nodes, communicate_edge_index):
    neighbor_layer = [[]]
    L_hop_nodes_copy = L_hop_nodes.clone()
    for hop in range(1, num_layers + 1):
        neighbors = set()
        for element in communicate_edge_index.T:
            if element[1] in L_hop_nodes:
                neighbors.add(element[0])

        L_hop_nodes = torch.tensor(list(neighbors)).tolist()

        neighbor_layer.append(torch.tensor(list(neighbors)))

    return neighbor_layer

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

    neighbor_layer = [[]] + copy.deepcopy(subsets)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    return neighbor_layer

def new_federated_GCN_embedding_update_periodic_cluster(client_node_index_list, K, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN1(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()


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
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])


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
        # neighbors_layer = get_all_Lhop_neighbors(client_node_index_list[i].clone(), communicate_edge_index)
        neighbors_layer = get_all_Lhop_neighbors_new(client_node_index_list[i].clone(), edge_index, L_hop)
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
        client_adj_t_partial = remove_intra_client_edge_new(client_adj_t, in_com_train_nei_indexes[i][L_hop-1])
        client_adj_t_partial_list.append(client_adj_t_partial)

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
    models = []
    for i in range(K):
        models.append(GCN1(nfeat=features.shape[1],
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
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test], 'hop_federated_embedding_periodic_cluster_'+str(period)+'_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test




def federated_cluster_GCN_partial(dataset, data, K, features, adj, labels, idx_train,
                  idx_val, idx_test, iid_percent, L_hop, num_layers, participant_one_round):
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
    edge_index_full = data.edge_index  # 所有clusters都参与training的edge_index

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

    # row, col, edge_attr = adj.t().coo()
    # edge_index = torch.stack([row, col], dim=0)

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
        communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i], L_hop, data.edge_index)[0]

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

    # participant_one_round = 5
    for t in range(global_epoch):
        edge_index = edge_index_full
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print(client_participant_indexes)
        # nodes_not_partici_training = torch.tensor([])
        # for i in range(K):
        #     if i in client_participant_indexes:
        #         continue
        #     nodes_not_partici_training = torch.cat([nodes_not_partici_training, split_data_indexes[i]], 0)
        # nodes_not_partici_training = nodes_not_partici_training.sort()[0]
        # edge_index_new = remove_not_training_nodes(edge_index, nodes_not_partici_training)
        # edge_index = edge_index_new

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        # for i in range(K):
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
                acc_train = Lhop_Block_matrix_train(iteration, models[i], optimizers[i],
                                                    features, adj, labels, communicate_indexes[i],
                                                    in_com_train_data_indexes[i])

            acc_trains.append(acc_train)  # 保存loss和accuracy

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
                     'hop_'+str(participant_one_round)+'_participants_federated_cluster_partial_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def federated_cluster_GCN(dataset, data, K, features, adj, labels,
                          idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
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
    edge_index = data.edge_index


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

    # row, col, edge_attr = adj.t().coo()
    # edge_index = torch.stack([row, col], dim=0)

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

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test], 'hop_federated_cluster_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

# federated GCN, 不传输node features，而是传输embeddings
def federated_GCN_embedding(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN1(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()


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
    client_adj_t_partial_list = []
    in_com_train_nei_indexes = []
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # 返回split_data_indexes[i]+目标节点为split_data_indexes[i]的L_hop nodes组成的子图（即nodes set+edge set）
        # 这里[0]获取的是nodes集合，[1]：edge_index,[2]：貌似是[0]中nodes在当前子图中的index,
        # [3]：Ture的index为当前子图edge在全图edge_index的index
        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
                                                                 L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(split_data_indexes[i], idx_train)

        # torch.searchsorted(sortedsequence=[1,3,5,7,9], values=[3,6,9])返回一个和values一样大小的tensor,
        # 其中的元素是在sorted_sequence中大于等于values中值的索引
        # 本例中输出为[1,3,4]，其中1，4分别是value=3，9的index，而3则是(7>=6)得到的
        # 因为communicate_indexes[i]包含split_data_indexes[i]，而split_data_indexes[i]包含inter，
        # 所以这里返回的是精准查找的结果，即communicate_indexes[i]中出现inter的index
        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors(split_data_indexes[i].clone(), communicate_edge_index)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = split_data_indexes[i].clone()
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], split_data_indexes[i])

            # ####这部分是之前写错的，但是用k_hop_subgraph或者的1-hop nodes，去掉local nodes，得到的理应
            # ####是L-1 layer的cross-client neighbors，数量和get_all_Lhop_neighbors()计算的不一致
            # split_data_indexes[i]的所有layer-hop neighbors（包括split_data_indexes[i]）
            # all_nodes_layer = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
            #                                                        hop, edge_index)[0]
            # split_data_indexes[i]的第layer层neighbor, 去掉了在本地的neighbors
            # cross_client_neighbor = setdiff1d(all_nodes_layer, all_nodes_layer_before).sort()[0]
            # all_nodes_layer_before = all_nodes_layer.clone()
            # #####后面结果不对的时候再用这段代码检查一下

            # cross_client_neighbor_list.append(cross_client_neighbor)
            cross_client_neighbor_list.append(
                torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

        in_com_train_nei_indexes.append(cross_client_neighbor_list)
        # client i上的邻接矩阵
        client_adj_t = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 用于计算(L-1)-layer上的nodes的emb.，即去掉L-layer的cross-client neighbors相关edges
        client_adj_t_partial = remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop-1])
        client_adj_t_partial_list.append(client_adj_t_partial)

    # ###以下为local training
    # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
    # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
    models = []
    for i in range(K):
        models.append(GCN1(nfeat=features.shape[1],
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
                acc_train = train_histories(iteration, models[i], optimizers[i],
                                            features, adj, labels, communicate_indexes[i],
                                            in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                            in_com_train_nei_indexes[i])

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
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test], 'hop_federated_embedding_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


def federated_GCN_embedding_update_realtime(K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN1(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()


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
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # 返回split_data_indexes[i]+目标节点为split_data_indexes[i]的L_hop nodes组成的子图（即nodes set+edge set）
        # 这里[0]获取的是nodes集合，[1]：edge_index,[2]：貌似是[0]中nodes在当前子图中的index,
        # [3]：Ture的index为当前子图edge在全图edge_index的index
        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
                                                                 L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(split_data_indexes[i], idx_train)

        # torch.searchsorted(sortedsequence=[1,3,5,7,9], values=[3,6,9])返回一个和values一样大小的tensor,
        # 其中的元素是在sorted_sequence中大于等于values中值的索引
        # 本例中输出为[1,3,4]，其中1，4分别是value=3，9的index，而3则是(7>=6)得到的
        # 因为communicate_indexes[i]包含split_data_indexes[i]，而split_data_indexes[i]包含inter，
        # 所以这里返回的是精准查找的结果，即communicate_indexes[i]中出现inter的index
        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], split_data_indexes[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors(split_data_indexes[i].clone(), communicate_edge_index)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = split_data_indexes[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], split_data_indexes[i])

            # ####这部分是之前写错的，但是用k_hop_subgraph或者的1-hop nodes，去掉local nodes，得到的理应
            # ####是L-1 layer的cross-client neighbors，数量和get_all_Lhop_neighbors()计算的不一致
            # split_data_indexes[i]的所有layer-hop neighbors（包括split_data_indexes[i]）
            # all_nodes_layer = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
            #                                                        hop, edge_index)[0]
            # split_data_indexes[i]的第layer层neighbor, 去掉了在本地的neighbors
            # cross_client_neighbor = setdiff1d(all_nodes_layer, all_nodes_layer_before).sort()[0]
            # all_nodes_layer_before = all_nodes_layer.clone()
            # #####后面结果不对的时候再用这段代码检查一下

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
    for i in range(K):
        models.append(GCN1(nfeat=features.shape[1],
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
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new(t, models[i], optimizers[i],
                                            features, adj, labels, communicate_indexes[i],
                                            in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                            in_com_train_nei_indexes[i], in_data_nei_indexes[i], split_data_indexes[i],
                                            in_com_train_local_node_indexes[i], global_model)

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
        loss_train, acc_train = test_hist(global_model, features, adj, labels, idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist(global_model, features, adj, labels, idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist(global_model, features, adj, labels, idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test], 'hop_federated_embedding_realtime_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def federated_GCN_embedding_update_periodic(K, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN1(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()


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
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
        split_data_indexes[i] = split_data_indexes[i].sort()[0]

        # 返回split_data_indexes[i]+目标节点为split_data_indexes[i]的L_hop nodes组成的子图（即nodes set+edge set）
        # 这里[0]获取的是nodes集合，[1]：edge_index,[2]：貌似是[0]中nodes在当前子图中的index,
        # [3]：Ture的index为当前子图edge在全图edge_index的index
        communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
                                                                 L_hop, edge_index)[0: 2]

        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        # 训练集idx_train中client i所包含的nodes集合
        inter = intersect1d(split_data_indexes[i], idx_train)

        # torch.searchsorted(sortedsequence=[1,3,5,7,9], values=[3,6,9])返回一个和values一样大小的tensor,
        # 其中的元素是在sorted_sequence中大于等于values中值的索引
        # 本例中输出为[1,3,4]，其中1，4分别是value=3，9的index，而3则是(7>=6)得到的
        # 因为communicate_indexes[i]包含split_data_indexes[i]，而split_data_indexes[i]包含inter，
        # 所以这里返回的是精准查找的结果，即communicate_indexes[i]中出现inter的index
        in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        in_com_train_local_node_indexes.append(
            torch.searchsorted(communicate_indexes[i], split_data_indexes[i]).clone())
        # 分层保存所有neighbors
        neighbors_layer = get_all_Lhop_neighbors(split_data_indexes[i].clone(), communicate_edge_index)
        # 分层保存cross-client neighbors，共L_hop layers
        cross_client_neighbor_list = []
        # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
        all_nodes_layer_before = split_data_indexes[i].clone()
        all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
        for hop in range(1, L_hop + 1):
            cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], split_data_indexes[i])

            # ####这部分是之前写错的，但是用k_hop_subgraph或者的1-hop nodes，去掉local nodes，得到的理应
            # ####是L-1 layer的cross-client neighbors，数量和get_all_Lhop_neighbors()计算的不一致
            # split_data_indexes[i]的所有layer-hop neighbors（包括split_data_indexes[i]）
            # all_nodes_layer = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
            #                                                        hop, edge_index)[0]
            # split_data_indexes[i]的第layer层neighbor, 去掉了在本地的neighbors
            # cross_client_neighbor = setdiff1d(all_nodes_layer, all_nodes_layer_before).sort()[0]
            # all_nodes_layer_before = all_nodes_layer.clone()
            # #####后面结果不对的时候再用这段代码检查一下

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
    for i in range(K):
        models.append(GCN1(nfeat=features.shape[1],
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
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, models[i], optimizers[i],
                                            features, adj, labels, communicate_indexes[i],
                                            in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                            in_com_train_nei_indexes[i], in_data_nei_indexes[i], split_data_indexes[i],
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
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test], 'hop_federated_embedding_periodic_'+str(period)+'_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test


def federated_GCN_embedding_update_periodic_cluster(client_node_index_list, K, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period):
    # K: number of local models/clients
    # choose adj matrix
    # multilayer_GCN:n*n
    # define model
    global_model = GCN1(nfeat=features.shape[1],
                       nhid=args_hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args_dropout,
                       NumLayers=num_layers,
                       num_nodes = data.num_nodes).to(device)
    global_model.reset_parameters()


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
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])


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
        neighbors_layer = get_all_Lhop_neighbors(client_node_index_list[i].clone(), communicate_edge_index)
        # neighbors_layer = get_all_Lhop_neighbors(client_node_index_list[i].clone(), edge_index)
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
    for i in range(K):
        models.append(GCN1(nfeat=features.shape[1],
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
                     [loss_train, acc_train,loss_val, acc_val, loss_test, acc_test], 'hop_federated_embedding_periodic_cluster_'+str(period)+'_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test

def federated_GCN_embedding_update_periodic_cluster_update(client_node_index_list,
                                            cluster_partition, cluster_update_period,
                                            K, features, adj, labels,
                                            idx_train, idx_val, idx_test,
                                            iid_percent, L_hop, num_layers, period):
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

    # 将cluster的更新period相同的nodes合并到一起
    cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
    for i in range(len(cluster_partition)):
        cluster_nodes = cluster_partition[i]
        cluster_partition_group[cluster_update_period[i]] += cluster_nodes

    [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

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
    client_adj_t_partial_list = []
    in_data_nei_indexes = []
    in_com_train_nei_indexes = []
    in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
    for i in range(K):
        if args_cuda:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i]).cuda()
        else:
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])


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
        neighbors_layer = get_all_Lhop_neighbors(client_node_index_list[i].clone(), communicate_edge_index)
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

    for i in range(K):
        models[i].load_state_dict(global_model.state_dict())

    for t in range(global_epoch):
        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        for index in range(len(cluster_partition_group)):
            if cluster_partition_group[index] != []:
                if t >= 0 and (t + 1) % index == 0:
                    global_model.update_hists_cluster(cluster_partition_group[index])

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

        write_result(t, iid_percent, L_hop, K,
                     [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test], 'hop_Block_federated_')

        for i in range(K):
            models[i].load_state_dict(gloabl_state)

    del global_model, features, adj, labels, idx_train, idx_val, idx_test
    while len(models) >= 1:
        del models[0]

    return loss_test, acc_test



# %%
def federated_GCN_cluster(client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, iid_percent, L_hop, num_layers):
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

def federated_GCN_cluster_partial(data, client_node_index_list, K, features, adj, labels,
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
        client_participant_indexes = random.sample(range(0, K), participant_one_round)
        client_participant_indexes.sort()
        print(client_participant_indexes)

        client_participant_indexes = [13, 18, 9, 16, 11]
        # 1.2 local training，每一次迭代local_iteration次
        # for i in range(K):
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
                acc_train = Lhop_Block_matrix_train(iteration, models[i], optimizers[i],
                                                    features, adj, labels, communicate_indexes[i],
                                                    in_com_train_data_indexes[i])

            # acc_trains.append(acc_train)  # 保存loss和accuracy

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
    features, adj, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    # dataset, data, features, adj, labels, idx_train, idx_val, idx_test = load_data_new(dataset_name)
    class_num = labels.max().item() + 1  # 因为labels编号0-6
    # data.adj_t = gcn_norm(data.adj_t)

# %%
if dataset_name in ['simulate', 'cora', 'citeseer', 'pubmed']:
    args_hidden = 16
else:
    args_hidden = 256

args_dropout = 0.5
args_lr = 0.1
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
    if f.find("cora_IID") >= 0:
        os.remove(path + f)

if dataset_name in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root="data/", name=dataset_name)
else:
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0]
K = 20  # client_num
client_node_index_list, cluster_partition, cluster_update_period, \
            client_cluster_nodes = partition_data_for_client(data, K, 10, '')

# centralized_cluster_GCN_mini(dataset, data, features, adj, labels, idx_train, idx_val, idx_test, num_layers)
# 集中式GCN
# centralized_GCN(features, adj, labels, idx_train, idx_val, idx_test, num_layers)

# 集中式GCN + 去掉intra-cluster edges+部分客户端选择
# centralized_cluster_GCN(data, features, adj, labels, idx_train, idx_val, idx_test, num_layers)

# FedGCN without cross-client communication + 基于cluster的数据划分
# federated_GCN_cluster(client_node_index_list, class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          0, num_layers)

# 每个epoch选不同的clients参与training
# federated_GCN_cluster_partial(data, client_node_index_list, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers,
#                               participant_one_round=5)
# federated_GCN_cluster_partial(data, client_node_index_list, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers,
#                               participant_one_round=5)
# federated_GCN_cluster_partial(data, client_node_index_list, K, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers,
#                               participant_one_round=K)

# 不同clusters设置不同更新频率，频率list为cluster_update_period
# federated_GCN_embedding_update_periodic_cluster_update(client_node_index_list, cluster_partition, cluster_update_period,
#                         class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=1)

# 所有clusters设置同一更新频率 + 基于cluster的数据划分
# new_federated_GCN_embedding_update_periodic_cluster(client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=1)
# federated_GCN_cluster(client_node_index_list, K, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          0, num_layers)
# federated_GCN_embedding_update_periodic_cluster(client_node_index_list, class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=1)
# federated_GCN_embedding_update_periodic_cluster(client_node_index_list, class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=5)
# federated_GCN_embedding_update_periodic_cluster(client_node_index_list, class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=25)
# federated_GCN_embedding_update_periodic_cluster(client_node_index_list, class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=40)

# 所有clusters设置同一更新频率 + 原始数据划分
# federated_GCN_embedding_update_realtime(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers)
# federated_GCN_embedding_update_periodic(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=1)
# federated_GCN_embedding_update_periodic(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=10)
# federated_GCN_embedding_update_periodic(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          num_layers, num_layers, period=50)

# 原始数据划分方式+去掉intra-cluster edges+部分客户端选择
# federated_cluster_GCN_partial(dataset, data, class_num, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers, participant_one_round=3)
# federated_cluster_GCN_partial(dataset, data, class_num, features, adj, labels,
#                       idx_train, idx_val, idx_test, 1, 0, num_layers, participant_one_round=5)
# federated_cluster_GCN(dataset, data, class_num, features, adj, labels,
#                           idx_train, idx_val, idx_test, 1, 0, num_layers)

# 原始数据划分方式
# Lhop_Block_federated_GCN(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          0, num_layers)  # 0-hop neighbor communication
#
# Lhop_Block_federated_GCN(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          1, num_layers)  # 1-hop neighbor communication
# Lhop_Block_federated_GCN(class_num, features, adj, labels, idx_train, idx_val, idx_test, 1,
#                          2, num_layers)
BDS_federated_GCN(class_num, features, adj, labels, idx_train, idx_val, idx_test, 0)

