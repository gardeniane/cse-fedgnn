# setting of data generation

import torch
import random
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx
import torch_geometric
import torch_sparse
from torch_geometric.datasets import Planetoid
import itertools
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

def generate_data(number_of_nodes, class_num, link_inclass_prob, link_outclass_prob):

    adj = torch.zeros(number_of_nodes,number_of_nodes)  # n*n adj matrix

    labels = torch.randint(0,class_num,(number_of_nodes,))  # assign random label with equal probability
    labels = labels.to(dtype=torch.long)
    # label_node, speed up the generation of edges
    label_node_dict=dict()

    for j in range(class_num):
            label_node_dict[j]=[]

    for i in range(len(labels)):
        label_node_dict[int(labels[i])]+=[int(i)]

    # generate graph
    for node_id in range(number_of_nodes):
                j=labels[node_id]
                for l in label_node_dict:
                    if l==j:
                        for z in label_node_dict[l]:  # z>node_id,  symmetrix matrix, no repeat
                            if z>node_id and random.random()<link_inclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                    else:
                        for z in label_node_dict[l]:
                            if z>node_id and random.random()<link_outclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                              
    adj = torch_geometric.utils.dense_to_sparse(torch.tensor(adj))[0]

    # generate feature use eye matrix
    features=torch.eye(number_of_nodes,number_of_nodes)

    # seprate train,val,test
    idx_train = torch.LongTensor(range(number_of_nodes//5))
    idx_val = torch.LongTensor(range(number_of_nodes//5, number_of_nodes//2))
    idx_test = torch.LongTensor(range(number_of_nodes//2, number_of_nodes))

    return features.float(), adj, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        number_of_nodes=adj.shape[0]


        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        #features = normalize(features) #cannot converge if use SGD, why??????????
        #adj = normalize(adj)    # no normalize adj here, normalize it in the training process


        features=torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        #edge_index=torch_geometric.utils.dense_to_sparse(torch.tensor(adj.toarray()))[0]
        labels=torch.tensor(labels)
        labels=torch.argmax(labels,dim=1)
    elif dataset_str in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-papers100M']: #'ogbn-mag' is heteregeneous
        #from ogb.nodeproppred import NodePropPredDataset
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        #dataset = NodePropPredDataset(name = dataset_str, root = 'dataset/')
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=torch_geometric.transforms.ToSparseTensor())

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]
        
        features = data.x #torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1) #torch.tensor(graph[1].reshape(-1))
        adj = data.adj_t.to_symmetric()
        #edge_index = torch.tensor(graph[0]['edge_index'])
        #adj = torch_geometric.utils.to_dense_adj(torch.tensor(graph[0]['edge_index']))[0]
    if dataset_str == 'citeseer':
        return features.float(), adj, labels, idx_train, idx_val, idx_test, graph
    return features.float(), adj, labels, idx_train, idx_val, idx_test


def load_data_new(dataset_str):
    if dataset_str in ['cora', 'citeseer', 'pubmed']:

        dataset = Planetoid(root="data/", name=dataset_str)
        data = dataset[0]

        features = data.x  # torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1)  # torch.tensor(graph[1].reshape(-1))
        nodes_index = torch.tensor(range(data.num_nodes))
        idx_train = nodes_index[data.train_mask]
        idx_val = nodes_index[data.val_mask]
        idx_test = nodes_index[data.test_mask]
        edge_list = data.edge_index.t().tolist()
        # 构造邻接矩阵
        g = nx.MultiGraph()  # 无向多边图，即一对nodes允许存在多条边
        # itertools.chain(*edge_index)将edge_index去掉维度，拼接成一个扁平的list（一行）
        # set()去除重复元素，然后再用sorted()排序
        nodeset = sorted(set(itertools.chain(*edge_list)))
        # 添加nodes、edges，然后生成邻接矩阵
        g.add_nodes_from(nodeset)
        g.add_edges_from(edge_list)
        adj = sp.lil_matrix(nx.adjacency_matrix(g))
        adj = torch.tensor(adj.toarray())
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])

        # transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        # dataset = Planetoid(root="data/", name=dataset_str, transform=transform)
        # data = dataset[0]
        # features = data.x  # torch.tensor(graph[0]['node_feat'])
        # labels = data.y.reshape(-1)  # torch.tensor(graph[1].reshape(-1))
        # nodes_index = torch.tensor(range(data.num_nodes))
        # idx_train = nodes_index[data.train_mask]
        # idx_val = nodes_index[data.val_mask]
        # idx_test = nodes_index[data.test_mask]
        # adj = data.adj_t

    elif dataset_str in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                         transform=torch_geometric.transforms.ToSparseTensor())

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]

        features = data.x  # torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1)  # torch.tensor(graph[1].reshape(-1))
        adj = data.adj_t.to_symmetric()
        row, col, edge_attr = adj.t().coo()
        edge_index = torch.stack([row, col], dim=0).T.tolist()
        data.edge_index = torch.tensor(edge_index).T
        data.num_nodes = features.shape[0]
        data.num_edges = data.edge_index.shape[1]
        dataset.data = data

    # 普通矩阵,n*n的形式
    # adj = torch.tensor(sp.lil_matrix(nx.adjacency_matrix(g)).todense())
    return dataset, data, features.float(), adj, labels, idx_train, idx_val, idx_test


from torch_geometric.loader import ClusterData


# partition_mode: 'uniform'——按照node_index顺序将nodes平均分配clients
def partition_data_for_client(data, client_num, cluster_num, partition_mode):
    cluster_partition, cluster_update_period = [], []
    cluster_data = ClusterData(data, num_parts=cluster_num)
    # 每两个相邻元素组成的区间（左开右闭）即为该cluster的nodes在cluster_data.perm中的indexs
    cluster_partptr = cluster_data.partptr
    cluster_perm = cluster_data.perm

    update_period = [1, 5, 25, 40]
    # 记录每个cluster的nodes
    for c in range(cluster_num):
        start = cluster_partptr[c]
        end = cluster_partptr[c+1]
        cluster_partition.append(cluster_perm[start:end].tolist())
        # 随机生成[5,10]区间内的一个整数
        # cluster_update_period.append(random.randint(5, 10))
        cluster_update_period.append(random.choice(update_period))

    average_cluster_node_share = []
    # 记录每个client从每个cluster分配到的平均nodes数目，最后一个client应该要稍微多一点
    [average_cluster_node_share.append((cluster_partptr[i+1] - cluster_partptr[i]) // client_num)
     for i in range(cluster_num)]
    client_node_index_list = []
    client_cluster_nodes = []  # shape为client_num*cluster_num
    for k in range(client_num):
        client_node_index = torch.IntTensor([])
        client_node_clusterwise = []  # 分cluster保存每个client的nodes
        for c in range(cluster_num):
            start = cluster_partptr[c] + k * average_cluster_node_share[c]
            if k == client_num - 1:
                end = cluster_partptr[c+1]
            else:
                end = cluster_partptr[c] + (k + 1) * average_cluster_node_share[c]
            client_node_index = torch.cat([client_node_index, cluster_perm[start:end]], 0)
            client_node_clusterwise.append(cluster_perm[start:end])

        client_cluster_nodes.append(client_node_clusterwise)

        client_node_index = client_node_index.sort()[0]
        client_node_index_list.append(client_node_index)

    return client_node_index_list, cluster_partition, cluster_update_period, client_cluster_nodes


class Data1(object):
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.edge_index = None
        self.adj = None