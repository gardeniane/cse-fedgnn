import logging
import torch
import data_process
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
import operator
from functools import reduce
import datetime

# 自定义库
from fl import FL
from train_func import test_local_hist, train_histories_new1, test_hist1
from fedGCN_embedding.history import History
from sklearn.decomposition import PCA


# 所有clusters设置不同通信period，即异步通信
class FL_HIST_ASYN(FL):
    def __init__(self, config):
        super().__init__(config)

        # 创建并初始化global model
        self.global_model = gcn2.GCN2(nfeat=self.features.shape[1],
                                    nhid=self.config.data.hidden_channels,
                                    nclass=self.labels.max().item() + 1,
                                    dropout=self.config.data.dropout,
                                    NumLayers=self.config.data.num_layers,
                                 num_nodes=self.data.num_nodes).to(self.device)
        self.global_model.reset_parameters()

        # 如果是FL mode，则创建并初始化clients的local models,否则为central training
        self.local_models, self.optimizers = [], []
        # 初始化local models，并设置optimizers等
        for i in range(config.clients.total):
            self.local_models.append(gcn2.GCN2(nfeat=self.features.shape[1],
                                               nhid=self.config.data.hidden_channels,
                                               nclass=self.labels.max().item() + 1,
                                               dropout=self.config.data.dropout,
                                               NumLayers=self.config.data.num_layers,
                                               num_nodes=self.data.num_nodes).to(self.device))

            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))

            self.local_models[i].load_state_dict(self.global_model.state_dict())

        # 为clients划分数据
        self.partition_data()

        self.period_max = self.config.fl.stages
        self.state_space = self.config.clients.total
        self.action_space = self.config.clusters * self.period_max

    def compute_comm_delay_cluster(self):  # 计算每个cluster的comm. time
        time_comm_client_cluster_list = []
        L_hop = self.config.data.num_layers
        for i in range(self.config.clients.total):
            time_comm_list = []  # 记录每个client的每一个cluster pull一次cross-client neighbors的时间
            for j in range(self.config.clusters):
                num_comm_emb = 0
                local_cluster_nodes = torch.tensor(self.client_cluster_nodes[i][j])
                # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
                local_cluster_nodes = local_cluster_nodes.sort()[0]

                neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_cluster_nodes[i][j].clone(),
                                                                  self.data.edge_index, L_hop)

                diff, diff_cluster = [], []
                index = 0
                for hop in range(1, L_hop):
                    diff.append(self.t2_minus_t1(local_cluster_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
                    num_comm_emb += diff[index].shape[0]

                time_comm_list.append(num_comm_emb * 4 * 16 * 1000 * 1000 / (1024 * 1024))

            time_comm_client_cluster_list.append(time_comm_list)

            self.time_comm_client_cluster_list = copy.deepcopy(time_comm_client_cluster_list)

    def compute_comm_delay(self):
        time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
        L_hop = self.config.data.num_layers
        for i in range(self.config.clients.total):
            num_comm_emb = 0
            local_nodes = torch.tensor(self.client_node_index_list[i])
            # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
            local_nodes = local_nodes.sort()[0]

            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                         self.data.edge_index, L_hop)

            diff, diff_cluster = [], []
            index = 0
            for hop in range(1, L_hop):
                diff.append(self.t2_minus_t1(local_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
                num_comm_emb += diff[index].shape[0]

            time_comm_list.append(num_comm_emb * 4 * 16 * 1000 * 1000 / (1024 * 1024))

        self.time_comm_list = copy.deepcopy(time_comm_list)

    # cluster-level的通信period设置
    def fgl_embed_cluster_level(self, cluster_update_period):
        config = self.config
        global_model = self.global_model

        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        self.dataset.data.edge_index = None
        self.dataset.data.edge_index = edge_index_new
        self.data = self.dataset[0]
        self.adj = self.edgeindex_to_adj(self.data.edge_index)

        K = config.clients.total
        L_hop = config.data.num_layers

        # 将cluster的更新period相同的nodes合并到一起
        cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
        for i in range(len(self.cluster_partition)):
            cluster_nodes = copy.copy(self.cluster_partition[i])
            cluster_partition_group[cluster_update_period[i]] += cluster_nodes

        [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

        # Train model

        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):

            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(self.client_node_index_list[i],
                                                                                             L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        # ###以下为local training
        # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
        # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
        for i in range(K):
            self.local_models.append(gcn2.GCN2(nfeat=self.features.shape[1],
                                     nhid=self.config.data.hidden_channels,
                                     nclass=self.labels.max().item() + 1,
                                     dropout=self.config.data.dropout,
                                     NumLayers=self.config.data.num_layers,
                                    num_nodes=communicate_indexes[i].shape[0]).to(self.device))

        for i in range(K):
            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))

        for i in range(K):
            self.local_models[i].load_state_dict(global_model.state_dict())

        for t in range(config.data.epochs):
            # 先根据cluster更新peridot更新cluster nodes的historical emb.
            for index in range(len(cluster_partition_group)):
                if cluster_partition_group[index] != []:
                    if t >= 0 and (t + 1) % index == 0:
                        global_model.update_hists_cluster(cluster_partition_group[index])

            acc_trains = []
            # 1.2 local training，每一次迭代local_iteration次
            for i in range(K):
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
                    # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                    acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                                     self.features, self.adj, self.labels, communicate_indexes[i],
                                                     in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                     in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                     self.client_node_index_list[i],
                                                     in_com_train_local_node_indexes[i], global_model, 0)

                acc_trains.append(acc_train)  # 保存loss和accuracy

            # 1.3 global aggregation
            states = []  # 保存clients的local models
            gloabl_state = dict()
            for i in range(K):
                states.append(self.local_models[i].state_dict())
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
            loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_train)
            print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

            loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_val)  # validation
            print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

            loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_test)
            print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

            self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                         '_fgl_embed_cluster_level_')

            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)

        del global_model, self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test
        while len(self.local_models) >= 1:
            del self.local_models[0]

        return loss_test, acc_test

    def make_client(self, L_hop):
        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        # self.dataset.data.edge_index = None
        # self.dataset.data.edge_index = edge_index_new
        # self.data = self.dataset[0]
        self.data.edge_index = edge_index_new
        self.adj = self.edgeindex_to_adj(self.data.edge_index.clone())

        K = self.config.clients.total

        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):

            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(
                self.client_node_index_list[i],
                L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)
        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.client_adj_t_partial_list = copy.deepcopy(client_adj_t_partial_list)
        self.in_com_train_nei_indexes = copy.deepcopy(in_com_train_nei_indexes)
        self.in_data_nei_indexes = copy.deepcopy(in_data_nei_indexes)
        self.in_com_train_local_node_indexes = copy.deepcopy(in_com_train_local_node_indexes)

        self.compute_comm_delay()
        self.compute_comm_delay_cluster()

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

        # ====以上，为clients划分local nodes，下面和make_client类似
        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):
            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(
                self.client_node_index_list[i],
                L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)
        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.client_adj_t_partial_list = copy.deepcopy(client_adj_t_partial_list)
        self.in_com_train_nei_indexes = copy.deepcopy(in_com_train_nei_indexes)
        self.in_data_nei_indexes = copy.deepcopy(in_data_nei_indexes)
        self.in_com_train_local_node_indexes = copy.deepcopy(in_com_train_local_node_indexes)

        self.compute_comm_delay()
        self.compute_comm_delay_cluster()


    def get_initiate_state(self):
        self.reset()
        self.client_comp_delay = [1] * 5 + [2] * 10 + [4] * 5  # 1/4 1s, 1/2 2s, 1/4 4s
        self.state = []
        for i in range(self.config.clients.total):
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                      self.labels, self.communicate_indexes[i],
                                      self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.state.append(local_loss)

        self.state = self.state + [max(self.client_comp_delay)]
        self.state.append(0)
        self.next_state = copy.copy(self.state)
        return copy.copy(self.state)

    def get_initiate_state_new(self):
        self.reset()
        self.state = []
        self.state = self.state + [0]*(self.config.clients.total+1)
        self.state.append(0)
        self.next_state = copy.copy(self.state)
        return copy.copy(self.state)

    def get_initiate_state_pca(self):
        self.reset()
        self.state = []
        initiate_state = self.get_state()
        self.next_state = copy.copy(self.state)
        return copy.copy(initiate_state)

    def get_state(self):
        pca = PCA(n_components=1)
        weights_list = []
        weights_list.append(self.extract_weights(self.global_model))
        for i in range(self.config.clients.total):
            weights_list.append(self.extract_weights(self.local_models[i]))
        state_i = []
        for j in range(len(weights_list)):
            for name_weight in weights_list[j]:
                if 'weight' in name_weight[0]:
                    reduce_x = pca.fit_transform(name_weight[1])
                    state_i = state_i + reduce_x.T.tolist()

        self.state = reduce(operator.add, state_i)
        return copy.copy(self.state)


    def extract_weights(self, model):
        weights = []
        for name, weight in model.to(torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                weights.append((name, weight.data))

        return weights

    def step_new(self, t, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        cluster_update_period = action[self.config.clients.per_round:]

        # 将cluster的更新period相同的nodes合并到一起
        cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
        for i in range(len(self.cluster_partition)):
            if cluster_update_period[i] == 0:
                continue
            cluster_nodes = copy.copy(self.cluster_partition[i])
            cluster_partition_group[cluster_update_period[i]] += cluster_nodes

        [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        if len(cluster_partition_group[0]) > 0:
            print("error")
        for index in range(1, len(cluster_partition_group)):
            if cluster_partition_group[index] != []:
                if t >= 0 and (t + 1) % index == 0:
                    global_model.update_hists_cluster(cluster_partition_group[index])

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        self.next_state = copy.copy(self.state)
        for i in range(K):
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                              self.features, self.adj, self.labels, self.communicate_indexes[i],
                              self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                              self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                              self.client_node_index_list[i],
                              self.in_com_train_local_node_indexes[i], global_model, 0)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = self.in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_embed_cluster_level_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # 计算local loss
        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss

        self.reward = - (self.config.data.target_accuracy - loss_train) * 10

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done

    def step(self, t, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        cluster_update_period = action[self.config.clients.per_round:]

        # 将cluster的更新period相同的nodes合并到一起
        cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
        for i in range(len(self.cluster_partition)):
            if cluster_update_period[i] == 0:
                continue
            cluster_nodes = copy.copy(self.cluster_partition[i])
            cluster_partition_group[cluster_update_period[i]] += cluster_nodes

        [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        if len(cluster_partition_group[0]) > 0:
            print("error")
        for index in range(1, len(cluster_partition_group)):
            if cluster_partition_group[index] != []:
                if t >= 0 and (t + 1) % index == 0:
                    global_model.update_hists_cluster(cluster_partition_group[index])

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        self.next_state = copy.copy(self.state)
        for i in range(K):
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                              self.features, self.adj, self.labels, self.communicate_indexes[i],
                              self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                              self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                              self.client_node_index_list[i],
                              self.in_com_train_local_node_indexes[i], global_model, 0)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = self.in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_embed_cluster_level_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # 计算local loss
        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss

        self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        # self.reward = self.get_reward(acc_test)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done

# 所有clients设置同一通信period
class FL_HIST_SYN(FL):
    def __init__(self, config):
        super().__init__(config)

        # 创建并初始化global model
        self.global_model = gcn.GCN1(nfeat=self.features.shape[1],
                                    nhid=self.config.data.hidden_channels,
                                    nclass=self.labels.max().item() + 1,
                                    dropout=self.config.data.dropout,
                                    NumLayers=self.config.data.num_layers,
                                 num_nodes=self.data.num_nodes).to(self.device)
        self.global_model.reset_parameters()

        self.local_models, self.optimizers = [], []
        # 为clients划分数据
        self.partition_data()

    # system-level的通信period设置
    def fgl_embed_system_level(self, period):
        config = self.config
        global_model = self.global_model

        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        self.dataset.data.edge_index = None
        self.dataset.data.edge_index = edge_index_new
        self.data = self.dataset[0]
        self.adj = self.edgeindex_to_adj(self.data.edge_index)

        K = config.clients.total
        L_hop = config.data.num_layers

        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):
            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(
                self.client_node_index_list[i],
                L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(
                # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i],
                                       torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge_new(client_adj_t,
                                                                in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        # ###以下为local training
        # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
        # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
        for i in range(K):
            self.local_models.append(gcn.GCN1(nfeat=self.features.shape[1],
                                   nhid=self.config.data.hidden_channels,
                                   nclass=self.labels.max().item() + 1,
                                   dropout=self.config.data.dropout,
                                   NumLayers=self.config.data.num_layers,
                               num_nodes=communicate_indexes[i].shape[0]).to(self.device))
            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))
            self.local_models[i].load_state_dict(self.global_model.state_dict())

        for t in range(config.data.epochs):
            acc_trains = []
            # 1.2 local training，每一次迭代local_iteration次
            for i in range(K):
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
                    # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                    acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                                     self.features, self.adj, self.labels, communicate_indexes[i],
                                                     in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                     in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                     self.client_node_index_list[i],
                                                     in_com_train_local_node_indexes[i], global_model, period)

                acc_trains.append(acc_train)  # 保存loss和accuracy

            # 1.3 global aggregation
            states = []  # 保存clients的local models
            gloabl_state = dict()
            for i in range(K):
                states.append(self.local_models[i].state_dict())
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
            loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_train)
            print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

            loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_val)  # validation
            print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

            loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_test)
            print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

            self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                         '_fgl_embed_system_level_period_' + str(period) + '_')

            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)

        del global_model, self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test
        while len(self.local_models) >= 1:
            del self.local_models[0]

        return loss_test, acc_test


# 用于每个clusters选不同clients交换emb.
class FL_HIST_PARTIAL(FL_HIST_ASYN):

    def __init__(self, config):
        super().__init__(config)

        self.state_space = self.config.clients.total + 1
        # 每个clusters选self.config.clients.per_round个clients
        self.action_space = self.config.clusters * self.config.clients.total


    # action由所有clusters的per_round clients组成
    def step(self, t, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total
        participant_clients = []

        # 先将action_list转变成cluster_client，action_list长度为clusters * clients.total
        # 实际长度为clusters * per_round
        # action_selected = []
        action.sort()
        for i in range(len(action)):
            c = action[i] // config.clients.total
            j = action[i] % config.clients.total
            # action_selected.append(j)
            global_model.update_hists_cluster(self.client_cluster_nodes[j][c])
            if j in participant_clients:
                continue
            else:
                participant_clients.append(j)

        # print("action_selected:", action_selected)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        self.state = copy.copy(self.next_state)
        max_comp_epoch = 0
        max_comm_epoch = 0
        for i in participant_clients:
            start_time = datetime.datetime.now()
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                              self.features, self.adj, self.labels, self.communicate_indexes[i],
                              self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                              self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                              self.client_node_index_list[i],
                              self.in_com_train_local_node_indexes[i], global_model, 0)

            end_time = datetime.datetime.now()
            update_delay = (end_time - start_time).microseconds
            if self.time_comm_list[i] >= max_comm_epoch:
                max_comm_epoch = self.time_comm_list[i]
            if update_delay >= max_comp_epoch:
                max_comp_epoch = update_delay

        self.computation_time.append(max_comp_epoch)
        self.communnication_time.append(max_comm_epoch)
        self.epoch_runtime.append(max_comp_epoch + max_comm_epoch)


        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in participant_clients:
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = participant_clients[0]
            gloabl_state[key] = self.in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(participant_clients)):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_embed_cluster_level_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # 计算local loss
        for i in participant_clients:
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss
        self.next_state[-1] = t
        # self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        # self.reward = self.get_reward(acc_test)
        # self.reward = self.get_reward_trade_partial(acc_test, max_comp_epoch + max_comm_epoch)
        self.reward = self.get_reward_time_only(max_comp_epoch + max_comm_epoch)
        print("runtime:", max_comp_epoch + max_comm_epoch, ", reward:", self.reward)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done

    # action由所有clusters的per_round clients组成
    def step_pca(self, t, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total
        participant_clients = []  # 本轮参与training的clients

        # 先将action_list转变成cluster_client，action_list长度为clusters * clients.total
        # 实际长度为clusters * per_round
        # action_selected = []
        action.sort()
        for i in range(len(action)):
            c = action[i] // config.clients.total
            j = action[i] % config.clients.total
            # action_selected.append(j)
            global_model.update_hists_cluster(self.client_cluster_nodes[j][c])
            if j in participant_clients:
                continue
            else:
                participant_clients.append(j)

        # print("action_selected:", action_selected)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        self.state = copy.copy(self.next_state)
        max_comp_epoch = 0
        max_comm_epoch = 0
        for i in participant_clients:
            start_time = datetime.datetime.now()
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                                 self.features, self.adj, self.labels, self.communicate_indexes[i],
                                                 self.in_com_train_data_indexes[i],
                                                 self.client_adj_t_partial_list[i],
                                                 self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                                                 self.client_node_index_list[i],
                                                 self.in_com_train_local_node_indexes[i], global_model, 0)
            end_time = datetime.datetime.now()
            update_delay = (end_time - start_time).microseconds
            if self.time_comm_list[i] >= max_comm_epoch:
                max_comm_epoch = self.time_comm_list[i]
            if update_delay >= max_comp_epoch:
                max_comp_epoch = update_delay

        self.computation_time.append(max_comp_epoch)
        self.communnication_time.append(max_comm_epoch)
        self.epoch_runtime.append(max_comp_epoch + max_comm_epoch)

            # acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in participant_clients:
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = participant_clients[0]
            gloabl_state[key] = self.in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(participant_clients)):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_embed_cluster_level_pca_')

        self.next_state = self.get_state()  # 在将global model赋值给clients之前，获取state

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # self.reward = - (self.config.data.target_accuracy - acc_train) * 10
        self.reward = self.get_reward(acc_test)
        # print("round:", t,", reward: ", self.reward)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done



class FL_HIST_STAGE_noCS(FL_HIST_ASYN):
    def __init__(self, config):
        super().__init__(config)

        # 创建并初始化global model
        self.global_model = gcn2.GCN3(nfeat=self.features.shape[1],
                                    nhid=self.config.data.hidden_channels,
                                    nclass=self.labels.max().item() + 1,
                                    dropout=self.config.data.dropout,
                                    NumLayers=self.config.data.num_layers,
                                 num_nodes=self.data.num_nodes).to(self.device)
        self.global_model.reset_parameters()

        # 如果是FL mode，则创建并初始化clients的local models,否则为central training
        self.local_models, self.optimizers = [], []
        # 初始化local models，并设置optimizers等
        for i in range(config.clients.total):
            self.local_models.append(gcn2.GCN3(nfeat=self.features.shape[1],
                                               nhid=self.config.data.hidden_channels,
                                               nclass=self.labels.max().item() + 1,
                                               dropout=self.config.data.dropout,
                                               NumLayers=self.config.data.num_layers,
                                               num_nodes=self.data.num_nodes).to(self.device))

            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))

            self.local_models[i].load_state_dict(self.global_model.state_dict())

        # 为clients划分数据
        self.partition_data()

        self.period_max = self.config.fl.stages
        self.state_space = self.config.clusters * 2  # clusters的loss、communication time、stage
        self.action_space = self.config.clusters * self.period_max

    def compute_comm_delay_cluster(self):  # 计算每个cluster的comm. time
        # cluster的总通信量（该cluster中每个client的inter-cluster neighbors的数量之和）*单个node embeddings的size/带宽
        time_comm_client_cluster_list = []
        L_hop = self.config.data.num_layers
        for i in range(self.config.clients.total):
            time_comm_list = []  # 记录每个client的每一个cluster pull一次cross-client neighbors的时间
            for j in range(self.config.clusters):
                num_comm_emb = 0
                local_cluster_nodes = torch.tensor(self.client_cluster_nodes[i][j])
                # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
                local_cluster_nodes = local_cluster_nodes.sort()[0]

                neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_cluster_nodes[i][j].clone(),
                                                                  self.data.edge_index, L_hop)

                diff, diff_cluster = [], []
                index = 0
                for hop in range(1, L_hop):
                    diff.append(self.t2_minus_t1(local_cluster_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
                    num_comm_emb += diff[index].shape[0]

                # 每个node embeddings共16维，每一维为4字节的float，client bandwidth为[1,10]Mbps
                time_comm_list.append(num_comm_emb * 4 * 16 / (3 * 1024))

            time_comm_client_cluster_list.append(time_comm_list)

        self.time_comm_client_cluster_list = copy.deepcopy(time_comm_client_cluster_list)

    def compute_comm_delay(self):
        time_comm_list = []  # 记录每个client pull一次cross-client neighbors的时间
        L_hop = self.config.data.num_layers
        for i in range(self.config.clients.total):
            num_comm_emb = 0
            local_nodes = torch.tensor(self.client_node_index_list[i])
            # .sort()顾名思义就是对tensor进行排序，排序之后和之前均保存了，排序序列为[0]，之前的为[1]
            local_nodes = local_nodes.sort()[0]

            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                         self.data.edge_index, L_hop)

            diff, diff_cluster = [], []
            index = 0
            for hop in range(1, L_hop):
                diff.append(self.t2_minus_t1(local_nodes, neighbors_layer[hop]))  # 需要communicated的nodes
                num_comm_emb += diff[index].shape[0]

            time_comm_list.append(num_comm_emb * 4 * 16 * 1000 * 1000 / (1024 * 1024))

        self.time_comm_list = copy.deepcopy(time_comm_list)

    # cluster-level的通信period设置
    def fgl_embed_cluster_level(self, cluster_update_period):
        config = self.config
        global_model = self.global_model

        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        self.dataset.data.edge_index = None
        self.dataset.data.edge_index = edge_index_new
        self.data = self.dataset[0]
        self.adj = self.edgeindex_to_adj(self.data.edge_index)

        K = config.clients.total
        L_hop = config.data.num_layers

        # 将cluster的更新period相同的nodes合并到一起
        cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
        for i in range(len(self.cluster_partition)):
            cluster_nodes = copy.copy(self.cluster_partition[i])
            cluster_partition_group[cluster_update_period[i]] += cluster_nodes

        [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

        # Train model

        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):

            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(self.client_node_index_list[i],
                                                                                             L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        # ###以下为local training
        # 1.1 assign global model weights to local models at initial step（奇怪，这个步骤为啥不放在global_epoch的循环里面
        # local models,这里最后一个参数num_nodes估计要调整为clients上的nodes数目
        for i in range(K):
            self.local_models.append(gcn2.GCN2(nfeat=self.features.shape[1],
                                     nhid=self.config.data.hidden_channels,
                                     nclass=self.labels.max().item() + 1,
                                     dropout=self.config.data.dropout,
                                     NumLayers=self.config.data.num_layers,
                                    num_nodes=communicate_indexes[i].shape[0]).to(self.device))

        for i in range(K):
            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))

        for i in range(K):
            self.local_models[i].load_state_dict(global_model.state_dict())

        for t in range(config.data.epochs):
            # 先根据cluster更新peridot更新cluster nodes的historical emb.
            for index in range(len(cluster_partition_group)):
                if cluster_partition_group[index] != []:
                    if t >= 0 and (t + 1) % index == 0:
                        global_model.update_hists_cluster(cluster_partition_group[index])

            acc_trains = []
            # 1.2 local training，每一次迭代local_iteration次
            for i in range(K):
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
                    # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                    acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                                     self.features, self.adj, self.labels, communicate_indexes[i],
                                                     in_com_train_data_indexes[i], client_adj_t_partial_list[i],
                                                     in_com_train_nei_indexes[i], in_data_nei_indexes[i],
                                                     self.client_node_index_list[i],
                                                     in_com_train_local_node_indexes[i], global_model, 0)

                acc_trains.append(acc_train)  # 保存loss和accuracy

            # 1.3 global aggregation
            states = []  # 保存clients的local models
            gloabl_state = dict()
            for i in range(K):
                states.append(self.local_models[i].state_dict())
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
            loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_train)
            print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

            loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_val)  # validation
            print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

            loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_test)
            print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

            self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                         '_fgl_embed_cluster_level_')

            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)

        del global_model, self.features, self.adj, self.labels, self.idx_train, self.idx_val, self.idx_test
        while len(self.local_models) >= 1:
            del self.local_models[0]

        return loss_test, acc_test

    def make_client(self, L_hop):
        edge_index_new = self.remove_crossclient_intracluster_edge(self.data.edge_index, self.cluster_partition, self.client_node_index_list)
        # self.dataset.data.edge_index = None
        # self.dataset.data.edge_index = edge_index_new
        # self.data = self.dataset[0]
        self.data.edge_index = edge_index_new
        self.adj = self.edgeindex_to_adj(self.data.edge_index.clone())

        K = self.config.clients.total

        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):

            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(
                self.client_node_index_list[i],
                L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)
        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.client_adj_t_partial_list = copy.deepcopy(client_adj_t_partial_list)
        self.in_com_train_nei_indexes = copy.deepcopy(in_com_train_nei_indexes)
        self.in_data_nei_indexes = copy.deepcopy(in_data_nei_indexes)
        self.in_com_train_local_node_indexes = copy.deepcopy(in_com_train_local_node_indexes)

        # self.compute_comm_delay()
        self.compute_comm_delay_cluster()

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

        # ====以上，为clients划分local nodes，下面和make_client类似
        communicate_indexes = []
        in_com_train_data_indexes = []
        client_adj_t_partial_list = []
        in_data_nei_indexes = []
        in_com_train_nei_indexes = []
        in_com_train_local_node_indexes = []  # clients的local nodes在communicate_indexes中的indexes
        for i in range(K):
            communicate_index, communicate_edge_index = torch_geometric.utils.k_hop_subgraph(
                self.client_node_index_list[i],
                L_hop, self.data.edge_index)[0: 2]

            communicate_indexes.append(communicate_index)
            communicate_indexes[i] = communicate_indexes[i].sort()[0]

            # 训练集idx_train中client i所包含的nodes集合
            inter = self.intersect1d(self.client_node_index_list[i], self.idx_train)

            in_com_train_data_indexes.append(  # in_com_train_data_indexes保存无需通信的nodes在communicate_indexes[i]的index集合
                torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix

            in_com_train_local_node_indexes.append(
                torch.searchsorted(communicate_indexes[i], self.client_node_index_list[i]).clone())
            # 分层保存所有neighbors
            neighbors_layer = self.get_all_Lhop_neighbors_new(self.client_node_index_list[i].clone(),
                                                              self.data.edge_index, L_hop)
            # 分层保存cross-client neighbors，共L_hop layers
            cross_client_neighbor_list = []
            # 分层保存cross-client neighbors在communicate_indexes[i]的indexes
            all_nodes_layer_before = self.client_node_index_list[i].clone()
            all_cross_client_neighbor = []  # 保存一个client的所有1-L-hop cross-client neighbors
            for hop in range(1, L_hop + 1):
                cross_client_neighbor = np.setdiff1d(neighbors_layer[hop], self.client_node_index_list[i])

                # 这里保存的是cross-client neighbors的IDs，即在data中的indexes
                all_cross_client_neighbor.append(torch.tensor(cross_client_neighbor).sort()[0])
                # 这里保存的是cross-client neighbors在communicate_indexes[i]中的indexes
                cross_client_neighbor_list.append(
                    torch.searchsorted(communicate_indexes[i], torch.tensor(cross_client_neighbor).sort()[0]).clone())

            in_data_nei_indexes.append(all_cross_client_neighbor)
            in_com_train_nei_indexes.append(cross_client_neighbor_list)
            # client i上的邻接矩阵
            client_adj_t = self.adj[communicate_indexes[i]][:, communicate_indexes[i]]
            # 用于计算(L-1)-hop上的nodes的emb.，即去掉L-hop的cross-client neighbors相关edges
            client_adj_t_partial = self.remove_intra_client_edge(client_adj_t, in_com_train_nei_indexes[i][L_hop - 1])
            client_adj_t_partial_list.append(client_adj_t_partial)

        self.in_com_train_data_indexes = copy.deepcopy(in_com_train_data_indexes)
        self.communicate_indexes = copy.deepcopy(communicate_indexes)
        self.client_adj_t_partial_list = copy.deepcopy(client_adj_t_partial_list)
        self.in_com_train_nei_indexes = copy.deepcopy(in_com_train_nei_indexes)
        self.in_data_nei_indexes = copy.deepcopy(in_data_nei_indexes)
        self.in_com_train_local_node_indexes = copy.deepcopy(in_com_train_local_node_indexes)

        self.compute_comm_delay()
        self.compute_comm_delay_cluster()

    def get_initiate_cluster_state(self):
        # self.state = [0.0] * (self.config.clusters*2)
        # self.state.append(0)
        # self.next_state = copy.deepcopy(self.state)
        # return copy.copy(self.state)
        state = []
        # state第一部分为global test loss
        for j in range(self.config.clusters):
            total_cluster_loss = 0.0
            total_client = self.config.clients.total
            for i in range(self.config.clients.total):
                if self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test).shape[0] == 0:
                    total_client -= 1
                    continue
                cluster_local_loss, cluster_local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                                        self.labels, self.communicate_indexes[i],
                                                                        self.intersect1d(
                                                                            self.client_cluster_nodes[i][j],
                                                                            self.idx_test))
                total_cluster_loss += cluster_local_loss
            state.append(total_cluster_loss/200)
        # 第二部分为total communication time
        for c in range(self.config.clusters):
            comm_time = sum(np.array(self.time_comm_client_cluster_list).T[c]) \
                        * (self.config.fl.stages // 1)
            state.append(comm_time/200)
        # 第三部分为当前stage
        # state.append(0)
        return copy.copy(state)

    def get_cluster_state(self, cluster_update_period, stage):
        state = []
        # state第一部分为global test loss
        for j in range(self.config.clusters):
            total_cluster_loss = 0.0
            total_client = self.config.clients.total
            for i in range(self.config.clients.total):
                if self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test).shape[0] == 0:
                    total_client -= 1
                    continue
                cluster_local_loss, cluster_local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                        self.labels, self.communicate_indexes[i],
                                                        self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test))
                total_cluster_loss += cluster_local_loss
            state.append(total_cluster_loss/200)
        # 第二部分为total communication time
        for c in range(self.config.clusters):
            comm_time = sum(np.array(self.time_comm_client_cluster_list).T[c]) \
                         * (self.config.fl.stages // cluster_update_period[c])
            state.append(comm_time/200)
        # 第三部分为当前stage
        # state.append(stage/(200/self.config.fl.stages))  # 注意200是MAX_EP_STEP的值
        return copy.copy(state)

    def get_initiate_state_new(self):
        self.reset()
        self.state = []
        self.state = self.state + [0]*(self.config.clients.total+1)
        self.state.append(0)
        self.next_state = copy.copy(self.state)
        return copy.copy(self.state)

    def extract_weights(self, model):
        weights = []
        for name, weight in model.to(torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                weights.append((name, weight.data))

        return weights

    def synchronize_hists(self, t, action_period):  # 每个stage为一个周期，t取值[0,self.config.fl.stages-1]
        for j in range(len(action_period)):
            # 先根据cluster更新peridot更新cluster nodes的historical emb.
            if action_period[j] == 1:
                for i in range(self.config.clients.total):
                    # 注意，只更新1-(L-1) hop的emb.
                    self.local_models[i].pull_latest_hists(self.global_model, self.in_data_nei_indexes[i][0])
            else:
                if (t + 1) % action_period[j] == 0:
                    for i in range(self.config.clients.total):
                        # 注意，只更新1-(L-1) hop的emb.
                        self.local_models[i].pull_latest_hists(self.global_model, self.in_data_nei_indexes[i][0])

    def step_new(self, stage, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        for t in range(self.config.fl.stages):
            self.synchronize_hists(t, action)
            for i in range(K):
                for iteration in range(config.fl.iterations):
                    if len(self.in_com_train_data_indexes[i]) == 0:
                        continue
                    try:
                        self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                    except:  # adj is empty
                        continue

                    # features, adj, labels等是整个dataset的数据
                    # 这里的communicate_indexes[i]是client i的training subgraph
                    # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                    # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                    acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                  self.features, self.adj, self.labels, self.communicate_indexes[i],
                                  self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                                  self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                                  self.client_node_index_list[i],
                                  self.in_com_train_local_node_indexes[i], global_model, 0)

                acc_trains.append(acc_train)  # 保存loss和accuracy

            # 1.3 global aggregation
            states = []  # 保存clients的local models
            gloabl_state = dict()
            for i in range(K):
                states.append(self.local_models[i].state_dict())
            # Average all parameters
            for key in global_model.state_dict():
                gloabl_state[key] = self.in_com_train_data_indexes[0].shape[0] * states[0][key]
                count_D = self.in_com_train_data_indexes[0].shape[0]
                for i in range(1, K):
                    gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                    count_D += self.in_com_train_data_indexes[i].shape[0]
                gloabl_state[key] /= count_D

            global_model.load_state_dict(gloabl_state)  # 更新global model
            # ###至此global aggregation结束

            # 1.4 Testing
            loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                               self.labels, self.idx_train)
            #
            # print(t+(stage-1)*self.config.fl.stages, '\t', "train", '\t', loss_train, '\t', acc_train)
            #
            # loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
            #                                self.labels, self.idx_val)  # validation
            # print(t+(stage-1)*self.config.fl.stages, '\t', "val", '\t', loss_val, '\t', acc_val)
            #
            loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                             self.labels, self.idx_test)
            # print(t+(stage-1)*self.config.fl.stages, '\t', "test", '\t', loss_test, '\t', acc_test)
            #
            # self.write_result(t+stage*self.config.fl.stages, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
            #                   '_fgl_embed_cluster_level_')

            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)

        # 1个stage结束
        # 计算cluster global loss
        next_state = self.get_cluster_state(action, stage)

        total_comm_time = 0
        for c in range(self.config.clusters):
            total_comm_time += sum(np.array(self.time_comm_client_cluster_list).T[c]) \
                         * (self.config.fl.stages // action[c])

        self.reward = - total_comm_time / 300

        return copy.copy(next_state), acc_test, self.reward, total_comm_time

    def step(self, t, action):
        config = self.config
        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        cluster_update_period = action[self.config.clients.per_round:]

        # 将cluster的更新period相同的nodes合并到一起
        cluster_partition_group = [[] for i in range(max(cluster_update_period) + 1)]
        for i in range(len(self.cluster_partition)):
            if cluster_update_period[i] == 0:
                continue
            cluster_nodes = copy.copy(self.cluster_partition[i])
            cluster_partition_group[cluster_update_period[i]] += cluster_nodes

        [cluster_partition_index.sort() for cluster_partition_index in cluster_partition_group]

        # 先根据cluster更新peridot更新cluster nodes的historical emb.
        if len(cluster_partition_group[0]) > 0:
            print("error")
        for index in range(1, len(cluster_partition_group)):
            if cluster_partition_group[index] != []:
                if t >= 0 and (t + 1) % index == 0:
                    global_model.update_hists_cluster(cluster_partition_group[index])

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        self.next_state = copy.copy(self.state)
        for i in range(K):
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                              self.features, self.adj, self.labels, self.communicate_indexes[i],
                              self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                              self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                              self.client_node_index_list[i],
                              self.in_com_train_local_node_indexes[i], global_model, 0)

            acc_trains.append(acc_train)  # 保存loss和accuracy

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = self.in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_embed_cluster_level_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # 计算local loss
        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss

        self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        # self.reward = self.get_reward(acc_test)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        return copy.copy(self.next_state), acc_test, self.reward, done



# 每个FL stage设置不同的communication interval，每个round选不同clients，而不是每个cluster选不同clients
class FL_HIST_PARTIAL_STAGE(FL_HIST_ASYN):

    def __init__(self, config):
        super().__init__(config)

        # 创建并初始化global model
        self.global_model = gcn2.GCN3(nfeat=self.features.shape[1],
                                      nhid=self.config.data.hidden_channels,
                                      nclass=self.labels.max().item() + 1,
                                      dropout=self.config.data.dropout,
                                      NumLayers=self.config.data.num_layers,
                                      num_nodes=self.data.num_nodes).to(self.device)
        self.global_model.reset_parameters()

        # 如果是FL mode，则创建并初始化clients的local models,否则为central training
        self.local_models, self.optimizers = [], []
        # 初始化local models，并设置optimizers等
        for i in range(config.clients.total):
            self.local_models.append(gcn2.GCN3(nfeat=self.features.shape[1],
                                               nhid=self.config.data.hidden_channels,
                                               nclass=self.labels.max().item() + 1,
                                               dropout=self.config.data.dropout,
                                               NumLayers=self.config.data.num_layers,
                                               num_nodes=self.data.num_nodes).to(self.device))

            self.optimizers.append(optim.SGD(self.local_models[i].parameters(),
                                             lr=self.config.data.lr,
                                             weight_decay=self.config.data.reg_weight_decay))

            self.local_models[i].load_state_dict(self.global_model.state_dict())

        # 用于client selection的agent的environment
        self.state_space = self.config.clients.total + 2
        # 每个clients选self.config.clients.per_round个clients
        self.action_space = self.config.clients.total

        # 用于communication interval setting的agent的environment
        self.val_state_space = self.config.clients.total + self.config.clusters + 2
        self.val_action_space = self.config.clusters * self.period_max

        self.val_state = list(np.zeros(self.val_state_space))
        self.val_next_state = self.val_state
        self.val_reward = -100
        self.val_action = []

        self.current_stage = 0  # 记录当前运行的stage num

        self.client_comp_delay = [10000]*5 + [20000]*10+[40000]*5

    def get_comm_one_stage(self, action_period):
        action_period.sort()
        cluster_update_period = []
        for i in range(len(action_period)):
            c = action_period[i] // self.period_max
            j = action_period[i] % self.period_max + 1
            # if j == 0:
            #     j = 1
            cluster_update_period.append(j)

        comm_time = 0
        for c in range(self.config.clusters):
            comm_time += max(np.array(self.time_comm_client_cluster_list).T[c]) \
                         * (self.config.fl.stages // cluster_update_period[c])

        self.communnication_time.append(comm_time)


    def synchronize_hists(self, t, action_period):
        action_period.sort()
        cluster_update_period = []
        for i in range(len(action_period)):
            c = action_period[i] // self.period_max
            j = action_period[i] % self.period_max + 1
            # if j == 0:
            #     j = 1
            cluster_update_period.append(j)

        self.cluster_update_period = cluster_update_period
        for j in range(len(cluster_update_period)):
            # 先根据cluster更新periodot更新cluster nodes的historical emb.
            if cluster_update_period[j] == 1:
                for i in range(self.config.clients.total):
                    # 注意，只更新1-(L-1) hop的emb.
                    self.local_models[i].pull_latest_hists(self.global_model, self.in_data_nei_indexes[i][0])
            else:
                stage_round = t % self.config.fl.stages
                if stage_round > 0 and stage_round % cluster_update_period[j] == 0:
                    for i in range(self.config.clients.total):
                        # 注意，只更新1-(L-1) hop的emb.
                        self.local_models[i].pull_latest_hists(self.global_model, self.in_data_nei_indexes[i][0])

    def get_inite_cluster_state(self):
        self.val_state = []
        for j in range(self.config.clusters):
            avg_cluster_loss = 0.0
            total_client = self.config.clients.total
            for i in range(self.config.clients.total):
                if self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test).shape[0] == 0:
                    total_client -= 1
                    continue
                cluster_local_loss, cluster_local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                        self.labels, self.communicate_indexes[i],
                                                        self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test))
                avg_cluster_loss += cluster_local_loss
            self.val_state.append(avg_cluster_loss/total_client)
        return copy.copy(self.val_state)

    def get_inite_cluster_state_loss(self):
        self.max_comm_time = 0
        K = self.config.clients.total
        C = len(self.cluster_update_period)
        self.val_state = copy.deepcopy(self.state[:K])
        for index in range(C):
            cluster_comm_time = self.cluster_total_comm_per_stage(index, 1)
            self.val_state.append(cluster_comm_time)
            self.max_comm_time += cluster_comm_time

        self.max_comm_time = 400
        for index in range(C):
            self.val_state[self.config.clients.total+index] = \
                self.val_state[self.config.clients.total+index] / self.max_comm_time
        self.val_state.append(max(self.val_state[K:K+C]))
        self.val_state.append(0)
        self.val_next_state = copy.deepcopy(self.val_state)
        return copy.copy(self.val_state)


    # 为每个cluster_index计算comm_interval对应的通信时间，一个stage
    def cluster_total_comm_per_stage(self, cluster_index, comm_interval):
        stages = self.config.fl.stages
        time_comm_client_cluster_arr = np.array(self.time_comm_client_cluster_list).T
        cluster_comm_time = sum(time_comm_client_cluster_arr[cluster_index])
        comm_time = cluster_comm_time * (stages // comm_interval - 1)
        return comm_time

    def update_cluster_state(self):
        for j in range(self.config.clusters):
            avg_cluster_loss = 0.0
            total_client = self.config.clients.total
            for i in range(self.config.clients.total):
                if self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test).shape[0] == 0:
                    total_client -= 1
                    continue
                cluster_local_loss, cluster_local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                        self.labels, self.communicate_indexes[i],
                                                        self.intersect1d(self.client_cluster_nodes[i][j], self.idx_test))
                avg_cluster_loss += cluster_local_loss / total_client
            self.val_next_state[j] += cluster_local_loss / self.config.clusters
        return copy.copy(self.val_next_state)

    def step(self, t, action):
        config = self.config

        global_model = self.global_model

        local_models = self.local_models
        optimizers = self.optimizers

        K = self.config.clients.total

        # 先将action_list转变成cluster_client，action_list长度为clusters * clients.total
        # 实际长度为clusters * per_round
        action_client = action[:self.config.clients.per_round]
        client_participant_indexes = copy.copy(action_client)

        # print("period:", self.cluster_update_period, ",action_client:",
        #       client_participant_indexes)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        # self.state = copy.copy(self.next_state)
        max_comp_epoch = 0
        max_comm_epoch = 0
        for i in client_participant_indexes:
            start_time = datetime.datetime.now()
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                              self.features, self.adj, self.labels, self.communicate_indexes[i],
                              self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                              self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                              self.client_node_index_list[i],
                              self.in_com_train_local_node_indexes[i], global_model, 0)

            # acc_trains.append(acc_train)  # 保存loss和accuracy
            end_time = datetime.datetime.now()
            update_delay = (end_time - start_time).microseconds
            # update_delay = self.client_comp_delay[i]
            # if self.time_comm_list[i] >= max_comm_epoch:
            #     max_comm_epoch = self.time_comm_list[i]
            if update_delay >= max_comp_epoch:
                max_comp_epoch = update_delay
            # if self.client_comp_delay[i] >= max_comp_epoch:
            #     max_comp_epoch = self.client_comp_delay[i]

        self.computation_time.append(max_comp_epoch)
        # self.communnication_time.append(max_comm_epoch)
        # self.epoch_runtime.append(max_comp_epoch + max_comm_epoch)

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in client_participant_indexes:
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            index = client_participant_indexes[0]
            gloabl_state[key] = self.in_com_train_data_indexes[index].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[index].shape[0]
            for i in range(1, len(client_participant_indexes)):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_interval_cs_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        # 计算local loss
        for i in range(K):
        # for i in client_participant_indexes:
            self.local_models[i].load_state_dict(gloabl_state)
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                               self.labels, self.communicate_indexes[i],
                                               self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.next_state[i] = local_loss

        self.next_state[K] = max_comp_epoch
        self.next_state[-1] = t
        # self.next_state = copy.deepcopy(self.next_state[K:])

        # self.reward = - (self.config.data.target_accuracy - loss_train) * 10
        self.reward = self.get_reward(acc_test)
        # self.reward = self.get_reward_trade_partial(acc_test, max_comp_epoch + max_comm_epoch)
        # self.reward = self.get_reward_time_only(max_comp_epoch)
        # self.reward = - max_comp_epoch / max(self.client_comp_delay)
        # print("runtime:", max_comp_epoch, ", reward:", self.reward)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        # 更新val_next_state
        if (t+1) % config.fl.stages == 0:  # 一个FL stage的最后一个round
            self.val_state = self.val_next_state
            self.val_next_state = list(np.zeros(self.val_state_space))
            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)
                local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                        self.labels, self.communicate_indexes[i],
                                                        self.intersect1d(self.client_node_index_list[i], self.idx_test))
                self.val_next_state[i] = local_loss

            for cluster_index in range(self.config.clusters):
                self.val_next_state[K+cluster_index] = self.cluster_total_comm_per_stage(cluster_index,
                                                            self.cluster_update_period[cluster_index]) / self.max_comm_time
            self.val_next_state[K+self.config.clusters] = max(self.val_state[K:K + self.config.clusters])
            self.val_next_state[-1] = t
            self.val_reward = self.get_reward_trade(acc_test)
            # print("val reward:", self.val_reward)

        return copy.copy(self.next_state), acc_test, self.reward, done


class FL_HIST_ALL_STAGE(FL_HIST_PARTIAL_STAGE):
    def __init__(self, config):
        super().__init__(config)

        self.state_space = self.config.clients.total + self.config.clusters + 2
        self.action_space = self.config.clusters * self.period_max
        self.max_comm_time = 400


    def get_initiate_state(self):
        self.reset()
        self.state = []
        for i in range(self.config.clients.total):
            local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                      self.labels, self.communicate_indexes[i],
                                      self.intersect1d(self.client_node_index_list[i], self.idx_test))
            self.state.append(local_loss)

        self.state = self.state + [0]*self.config.clusters  # communication delay of each client
        self.state.append(0)  # 最大的communication delay
        self.state.append(0)
        self.next_state = copy.copy(self.state)
        return copy.copy(self.state)

    def step(self, t):
        config = self.config

        global_model = self.global_model

        K = self.config.clients.total

        print("period:", self.cluster_update_period)

        acc_trains = []
        # 1.2 local training，每一次迭代local_iteration次
        # self.state = copy.copy(self.next_state)
        max_comp_epoch = 0
        for i in range(K):
            start_time = datetime.datetime.now()
            for iteration in range(config.fl.iterations):
                if len(self.in_com_train_data_indexes[i]) == 0:
                    continue
                try:
                    self.adj[self.communicate_indexes[i]][:, self.communicate_indexes[i]]
                except:  # adj is empty
                    continue

                # features, adj, labels等是整个dataset的数据
                # 这里的communicate_indexes[i]是client i的training subgraph
                # in_com_train_data_indexes[i]是client i的local nodes在communicate_indexes[i]中的index
                # client_adj_t_partial_list[i]是用于计算client i的L-hop node emb.的邻接矩阵

                acc_train = train_histories_new1(t, self.local_models[i], self.optimizers[i],
                                                 self.features, self.adj, self.labels, self.communicate_indexes[i],
                                                 self.in_com_train_data_indexes[i], self.client_adj_t_partial_list[i],
                                                 self.in_com_train_nei_indexes[i], self.in_data_nei_indexes[i],
                                                 self.client_node_index_list[i],
                                                 self.in_com_train_local_node_indexes[i], global_model, 0)

            # acc_trains.append(acc_train)  # 保存loss和accuracy
            end_time = datetime.datetime.now()
            update_delay = (end_time - start_time).microseconds
            if update_delay >= max_comp_epoch:
                max_comp_epoch = update_delay

        self.computation_time.append(max_comp_epoch)

        # 1.3 global aggregation
        states = []  # 保存clients的local models
        gloabl_state = dict()
        for i in range(K):
            states.append(self.local_models[i].state_dict())
        # Average all parameters
        for key in global_model.state_dict():
            gloabl_state[key] = self.in_com_train_data_indexes[0].shape[0] * states[0][key]
            count_D = self.in_com_train_data_indexes[0].shape[0]
            for i in range(1, K):
                gloabl_state[key] += self.in_com_train_data_indexes[i].shape[0] * states[i][key]
                count_D += self.in_com_train_data_indexes[i].shape[0]
            gloabl_state[key] /= count_D

        global_model.load_state_dict(gloabl_state)  # 更新global model
        # ###至此global aggregation结束

        # 1.4 Testing
        loss_train, acc_train = test_hist1(global_model, self.features, self.adj,
                                           self.labels, self.idx_train)
        print(t, '\t', "train", '\t', loss_train, '\t', acc_train)

        loss_val, acc_val = test_hist1(global_model, self.features, self.adj,
                                       self.labels, self.idx_val)  # validation
        print(t, '\t', "val", '\t', loss_val, '\t', acc_val)

        loss_test, acc_test = test_hist1(global_model, self.features, self.adj,
                                         self.labels, self.idx_test)
        print(t, '\t', "test", '\t', loss_test, '\t', acc_test)

        self.write_result(t, [loss_train, acc_train, loss_val, acc_val, loss_test, acc_test],
                          '_fgl_allstage_')

        for i in range(K):
            self.local_models[i].load_state_dict(gloabl_state)

        done = False
        if acc_test >= self.config.data.target_accuracy:
            done = True

        # 更新val_next_state
        if (t + 1) % config.fl.stages == 0:  # 一个FL stage的最后一个round
            self.state = copy.deepcopy(self.next_state)
            self.next_state = list(np.zeros(len(self.state)))
            for i in range(K):
                self.local_models[i].load_state_dict(gloabl_state)
                local_loss, local_acc = test_local_hist(self.local_models[i], self.features, self.adj,
                                                        self.labels, self.communicate_indexes[i],
                                                        self.intersect1d(self.client_node_index_list[i],
                                                                         self.idx_test))
                self.next_state[i] = local_loss
            for c in range(self.config.clusters):
                self.next_state[K + c] = max(np.array(self.time_comm_client_cluster_list).T[c])
            self.next_state[len(self.next_state) - 2] = max(self.time_comm_list)
            self.next_state[-1] = t
            self.reward = self.get_reward(acc_test) - self.next_state[len(self.next_state) - 2] / self.max_comm_time

        return acc_test, done