import logging
import torch
import data_process
from data_process import Data1
from models import gcn, gcn2
import copy
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric
from torch_geometric.loader import ClusterData
import torch.optim as optim
import random
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
import itertools
import torch_sparse
import matplotlib.pyplot as plt

# 自定义库
from train_func import test_local, Lhop_Block_matrix_train, test


class FL(object):
    def __init__(self, config):
        self.device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
        self.config = config
        # 加载数据集
        # if self.config.dataset_name in ['cora', 'citeseer', 'pubmed']:
        #     self.dataset = Planetoid(root="data/", name=self.config.dataset_name)
        #     self.data = self.dataset[0]
        # else:
        #     self.dataset = PygNodePropPredDataset(name='ogbn-arxiv')


        self.load_data()

        # RL相关
        self.state_space = 2  # self.config.clients.total
        self.action_space = self.config.clients.total

        # 用于保存计算和通信时间
        self.computation_time = []
        self.communnication_time = []
        self.epoch_runtime = []


    def load_data(self):
        if self.config.dataset_name == 'citeseer':
            self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test, graph \
                = data_process.load_data(self.config.dataset_name)
            self.data = Data1()
            edge_index = self.adj_dict2matrix(graph)
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack([row, col], dim=0).T.tolist()
            self.data.edge_index = torch.tensor(edge_index).T
            # data_obj.edge_index = data_obj.edge_index.to(device)
            self.data.num_nodes = self.features.shape[0]
            self.data.num_edges = self.data.edge_index.shape[1]
        else:
            self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test \
                = data_process.load_data(self.config.dataset_name)
            self.dataset = Planetoid(root="data/", name=self.config.dataset_name)
            self.data = self.dataset[0]

    def adj_dict2matrix(self, graph):
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        return adj

    def partition_data(self):
        if self.config.clients.data_partition == 'cluster-based':  # 先生成clusters再划分数据
            self.client_node_index_list, self.cluster_partition, self.cluster_update_period, \
            self.client_cluster_nodes = data_process.partition_data_for_client(self.data,
                                                       self.config.clients.total,
                                                       self.config.clusters, '')

    def normalize(self, mx):  # adj matrix

        mx = mx + torch.eye(mx.shape[0], mx.shape[1])

        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return torch.tensor(mx)

    def edgeindex_to_adj(self, edge_index):
        edge_list = edge_index.t().tolist()
        nodeset = sorted(set(itertools.chain(*edge_list)))
        g = nx.MultiGraph()  # 无向多边图，即一对nodes允许存在多条边
        g.add_nodes_from(nodeset)
        g.add_edges_from(edge_list)
        adj = sp.lil_matrix(nx.adjacency_matrix(g))
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        return adj

    # 删除cross-client edges
    def remove_intracluster_edge(self, edge_index, cluster_nodes):
        edge_index = edge_index.tolist()
        edge_num = len(edge_index[0])
        for i in range(edge_num - 1, -1, -1):
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
    def remove_crossclient_intracluster_edge(self, edge_index, cluster_nodes, client_node_index_list):
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

    # 删除不参与本轮training的nodes，通过删除其对应的edges
    def remove_not_training_nodes(self, edge_index, nodes_not_training):
        edge_index_copy = edge_index.clone()
        edge_index = edge_index.tolist()
        edge_num = len(edge_index[0])
        for i in range(edge_num - 1, -1, -1):
            if edge_index[0][i] in nodes_not_training or edge_index[1][i] in nodes_not_training:
                edge_index[0].pop(i)
                edge_index[1].pop(i)

        return torch.tensor(edge_index)

    # cross_client_neighbors: L-hop cross-client neighbors
    def remove_intra_client_edge(self, client_adj_t, cross_client_neighbors_indexes):
        row, col, edge_attr = client_adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0).T.tolist()

        for i in range(len(edge_index) - 1, -1, -1):
            if edge_index[i][0] in cross_client_neighbors_indexes \
                    or edge_index[i][1] in cross_client_neighbors_indexes:
                edge_index.pop(i)

        return torch.tensor(edge_index).T

    def remove_intra_client_edge_new(self, client_adj_t, cross_client_neighbors_indexes):
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
    def setdiff1d(self, t1, t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        # 计算过程应该是根据counts == 1得到index，然后再根据index取uniques的元素
        difference = uniques[counts == 1]
        return difference

    # 找到两个tensor中都出现的元素
    def intersect1d(self, t1, t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        # 计算过程应该是根据counts > 1得到index，然后再根据index取uniques的元素
        intersection = uniques[counts > 1]
        return intersection

    # 找到在t2但是不在t1中的元素
    def t2_minus_t1(self, t1, t2):
        difference = set(np.array(t2)) - set(np.array(t1))
        return torch.tensor(list(difference))

    # 将每个epoch的结果，即loss+accuracy写到文件中
    def write_result(self, epoch, one_round_result, filename_pre):
        loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = one_round_result
        filename = self.config.dataset_name + filename_pre + str(self.config.clients.per_round)
        # 先删除历史文件
        path = self.config.paths.model + "/"
        # model_files = os.listdir(path)
        # for i, f in enumerate(model_files):
        #     if f.find(filename) >= 0:
        #         os.remove(path + f)

        # 写文件
        a = open(path + filename, 'a+')

        a.write(str(epoch) + '\t' + "train" + '\t' + str(loss_train) + '\t' + str(acc_train) + '\n')
        a.write(str(epoch) + '\t' + "val" + '\t' + str(loss_val) + '\t' + str(acc_val) + '\n')
        a.write(str(epoch) + '\t' + "test" + '\t' + str(loss_test) + '\t' + str(acc_test) + '\n')
        a.close()
        # print("save file as", filename)

    # 获取node_idx的num_hops neighbors，分layer保存
    def get_all_Lhop_neighbors_new(self, node_idx, edge_index, num_hops):
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
            neighbor_layer.append(self.intersect1d(torch.tensor([], dtype=int), subsets[hop]))

        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]

        edge_index = edge_index[:, edge_mask]

        return neighbor_layer

    def reset(self):
        self.global_model.reset_parameters()
        for i in range(self.config.clients.total):
            self.local_models[i].reset_parameters()

        self.computation_time = []
        self.communnication_time = []
        self.epoch_runtime = []

    def get_reward(self, test_acc):
        return 128**(test_acc - self.config.data.target_accuracy) - 1

    def get_reward_trade_partial(self, test_acc, runtime):
        return (128**(test_acc - self.config.data.target_accuracy) - 1) * 0.5 \
               - runtime / 40000 * 0.5

    def get_reward_time_only(self, runtime):
        return - runtime / 40000

    def get_reward_trade(self, test_acc):
        stages = self.config.fl.stages
        time_comm_client_cluster_arr = np.array(self.time_comm_client_cluster_list).T
        comm_per_cluster = []
        max_comm_time_cluster = 0  # 一个cluster的最大通信时间
        for index in range(len(self.cluster_update_period)):
            cluster_period = self.cluster_update_period[index]
            cluster_comm_time = sum(time_comm_client_cluster_arr[index])
            comm_per_cluster.append(cluster_comm_time * (stages // cluster_period - 1))
            max_comm_time_cluster += cluster_comm_time * (stages - 1)
            if max_comm_time_cluster <= cluster_comm_time:
                max_comm_time_cluster = cluster_comm_time

        acc_reward = 128**(test_acc - self.config.data.target_accuracy) - 1
        # print("reward:", 128**(test_acc - self.config.data.target_accuracy), len(self.cluster_update_period), comm_num / (stages*len(self.cluster_update_period)))
        return - max(comm_per_cluster) / self.max_comm_time  # max_comm_time_cluster


    # def get_reward_comm(self):


    def plot_reward_result(self, x_vals, results_mean, mean_plus_std, mean_minus_std, dataset_name):
        ax = plt.gca()
        # ax.set_ylim([-30, 0])
        ax.set_ylabel('Reward', fontdict={'weight': 'normal', 'size': 28})
        ax.set_xlabel('Number of Episodes', fontdict={'weight': 'normal', 'size': 28})
        ax.plot(x_vals, results_mean, label='Average Result', color='blue')
        ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
        ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        # plt.legend([])
        plt.grid(True, which='major')
        plt.savefig('results/' + dataset_name + '_reward.pdf', format='pdf', bbox_inches='tight')
        plt.show()

class FL_PARTIAL(FL):
    def __init__(self, config):
        super().__init__(config)
        # 创建并初始化global model
        self.global_model = gcn.GCN(nfeat=self.features.shape[1],
                                    nhid=self.config.data.hidden_channels,
                                    nclass=self.labels.max().item() + 1,
                                    dropout=self.config.data.dropout,
                                    NumLayers=self.config.data.num_layers)
        self.global_model.reset_parameters()

        # 如果是FL mode，则创建并初始化clients的local models,否则为central training
        self.local_models, self.optimizers = [], []
        if self.config.mode == 'fl':
            for i in range(self.config.clients.total):
                self.local_models.append(gcn.GCN(nfeat=self.features.shape[1],
                                                 nhid=self.config.data.hidden_channels,
                                                 nclass=self.labels.max().item() + 1,
                                                 dropout=self.config.data.dropout,
                                                 NumLayers=self.config.data.num_layers))
                self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                                 lr=self.config.data.lr,
                                                 weight_decay=self.config.data.reg_weight_decay))
                self.local_models[i].load_state_dict(self.global_model.state_dict())

            # 为clients划分数据
            self.partition_data()

    def fgl_cluster_partial(self, L_hop):
        # L_hop: 是否与其他clients通信
        config = self.config
        global_model = self.global_model

        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        self.dataset.data.edge_index = None
        self.dataset.data.edge_index = edge_index_new
        self.data = self.dataset[0]
        self.adj = self.edgeindex_to_adj(self.data.edge_index.clone())

        local_models = self.local_models
        optimizers = self.optimizers
        K = config.clients.total

        communicate_indexes = []
        in_com_train_data_indexes = []
        for i in range(K):
            communicate_index = torch_geometric.utils.k_hop_subgraph(self.client_node_index_list[i],
                                                                     L_hop, self.data.edge_index)[0]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # only count the train data of nodes in current server(not communicate nodes)
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)  # 训练集idx_train中client i所包含的nodes集合

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        # ###以下为local training
        # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
        for i in range(K):
            local_models[i].load_state_dict(global_model.state_dict())

        for t in range(config.data.epochs):
            client_participant_indexes = random.sample(range(0, K), config.clients.per_round)
            client_participant_indexes.sort()
            acc_trains = []
            # 1.2 local training，每一次迭代local_iteration次
            # for i in range(K):
            for i in client_participant_indexes:
                acc_train = 0
                for iteration in range(config.fl.iterations):
                    if len(in_com_train_data_indexes[i]) == 0:
                        continue
                    try:
                        self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
                    except:  # adj is empty
                        continue

                    # features, adj, labels等是整个dataset的数据
                    # 这里的communicate_indexes[i]是client i的training subgraph
                    # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                    acc_train = Lhop_Block_matrix_train(iteration, local_models[i], optimizers[i],
                                                        self.features, self.data.edge_index, self.labels,
                                                        communicate_indexes[i], in_com_train_data_indexes[i])

                acc_trains.append(acc_train)  # 保存loss和accuracy

            # 1.3 global aggregation
            states = []  # 保存clients的local models
            gloabl_state = dict()
            # for i in range(K):
            for i in client_participant_indexes:
                states.append(local_models[i].state_dict())
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
            loss_train, acc_train = test(global_model, self.features, self.adj,
                                                    self.labels, self.idx_train)
            print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

            loss_val, acc_val = test(global_model, self.features, self.adj,
                                                self.labels, self.idx_val)  # validation
            print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

            loss_test, acc_test = test(global_model, self.features, self.adj,
                                                    self.labels, self.idx_test)
            print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

            self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                         '_fgl_cluster_partial_')

            for i in range(K):
                local_models[i].load_state_dict(gloabl_state)

        while len(local_models) >= 1:
            del local_models[0]

        return loss_test, acc_test

    def make_client(self, L_hop):
        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        self.dataset.data.edge_index = None
        self.dataset.data.edge_index = edge_index_new
        self.data = self.dataset[0]
        self.adj = self.edgeindex_to_adj(self.data.edge_index)

        K = self.config.clients.total
        communicate_indexes = []
        in_com_train_data_indexes = []
        for i in range(K):
            communicate_index = torch_geometric.utils.k_hop_subgraph(self.client_node_index_list[i],
                                                                     L_hop, self.data.edge_index)[0]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # only count the train data of nodes in current server(not communicate nodes)
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)  # 训练集idx_train中client i所包含的nodes集合

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)


    def make_client_iid(self, L_hop, iid_percent):
        self.client_node_index_list = []  # 保存local clients
        nclass = self.labels.max().item() + 1
        split_data_indexes = []
        non_iid_percent = 1 - float(iid_percent)
        iid_indexes = []  # random assign
        shuffle_labels = []  # make train data points split into different devices
        K = self.config.clients.total
        for i in range(K):
            current = torch.nonzero(self.labels == i).reshape(-1)
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
                (labels_class[
                 average_num * (i % average_device_of_class):average_num * (i % average_device_of_class + 1)]))

        iid_indexes = self.setdiff1d(torch.tensor(range(len(self.labels))), torch.cat(split_data_indexes))
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
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

            split_data_indexes[i] = split_data_indexes[i].sort()[0]
            self.client_node_index_list.append(split_data_indexes[i])
            # communicate_index=get_K_hop_neighbors(adj, split_data_indexes[i], L_hop) #normalized adj

            communicate_index = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],
                                      L_hop, self.data.edge_index)[0]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            inter = self.intersect1d(split_data_indexes[i],
                                self.idx_train)  ###only count the train data of nodes in current server(not communicate nodes)

            in_com_train_data_indexes.append(
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)

    def get_initiate_state(self):
        self.reset()
        state = []
        for i in range(self.config.clients.total):
            local_loss, local_acc = test_local(self.local_models[i], self.features, self.adj,
                                      self.labels, self.communicate_indexes[i],
                                      self.intersect1d(self.client_node_index_list[i], self.idx_test))
            state.append(local_loss)

        self.next_state = copy.deepcopy(state)
        return copy.copy(state)

    def get_initiate_state_delay(self):
        self.reset()
        self.client_comp_delay = [1] * 5 + [2] * 10 + [4] * 5  # 1/4 1s, 1/2 2s, 1/4 4s
        # state = [-1]*(self.config.clients.total + 1)
        # state = copy.deepcopy(self.client_comp_delay) + [0]
        state = [max(self.client_comp_delay),0]
        self.next_state = copy.deepcopy(state)
        self.state_space = len(state)
        return copy.copy(state)

    def step(self, t, action):
        # L_hop: 是否与其他clients通信
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        client_participant_indexes = action[:self.config.clients.per_round]
        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次

        for i in client_participant_indexes:
            acc_train = 0
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    print("client:", i)
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                acc_train = Lhop_Block_matrix_train(iteration, local_models[i], optimizers[i],
                                                    self.features, self.adj, self.labels,
                                                    self.communicate_indexes[i], self.in_com_train_data_indexes[i])


        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        # for i in range(K):
        for i in client_participant_indexes:
            states.append(local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = self.in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        self.global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test(global_model, self.features, self.adj,
                                                self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, self.features, self.adj,
                                            self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test(global_model, self.features, self.adj,
                                                self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                     '_fgl_cluster_partial_')

        # 计算local loss
        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss

        # self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        self.reward = self.get_reward(acc_test)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done

    def step_delay(self, t, action):
        # L_hop: 是否与其他clients通信
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        client_participant_indexes = action[:self.config.clients.per_round]
        print("client_participant_indexes:", client_participant_indexes)
        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        max_delay = 0
        for i in client_participant_indexes:
            if self.client_comp_delay[i] >= max_delay:
                max_delay = self.client_comp_delay[i]
        '''
            acc_train = 0
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    print("client:", i)
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                acc_train = Lhop_Block_matrix_train(iteration, local_models[i], optimizers[i],
                                                    self.features, self.adj, self.labels,
                                                    self.communicate_indexes[i], self.in_com_train_data_indexes[i])


        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        # for i in range(K):
        for i in client_participant_indexes:
            states.append(local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = self.in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        self.global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test(global_model, self.features, self.adj,
                                                self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test(global_model, self.features, self.adj,
                                            self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test(global_model, self.features, self.adj,
                                                self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                     '_fgl_cluster_partial_')

        # 计算local loss
        # for i in range(K):
        #     self.local_models[i].load_state_dict(gloabl_state)
        #     local_loss, local_acc = test_local(self.local_models[i], self.features, self.adj,
        #                                        self.labels, self.communicate_indexes[i],
        #                                        self.intersect1d(self.client_node_index_list[i], self.idx_test))
        #     self.next_state[i] = local_loss
        '''
        self.next_state = [max_delay,t]

        # self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        # self.reward = self.get_reward(acc_test)
        self.reward = - max_delay / max(self.client_comp_delay)
        print("reward:", self.reward)

        done = False
        acc_test = 0.0
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done