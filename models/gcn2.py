import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from fedGCN_embedding.history import History
from typing import Optional, Callable, Dict, Any
from torch.nn import ModuleList, BatchNorm1d


class GCN2(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers, num_nodes):
        super(GCN2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout
        self.drop_input = False
        self.linear = False

        self.num_nodes = num_nodes
        self.hidden_channels = nhid
        self.num_layers = NumLayers
        self.histories = torch.nn.ModuleList([
            History(num_nodes, nhid, device='cpu')
            for _ in range(NumLayers - 1)  # 分layer保存history embeddings
        ])

        self.histories_outdate = torch.nn.ModuleList([
            History(num_nodes, nhid, device='cpu')
            for _ in range(NumLayers - 1)  # 分layer保存history embeddings
        ])

        # self.bns = ModuleList()
        # for i in range(NumLayers):
        #     bn = BatchNorm1d(nhid)
        #     self.bns.append(bn)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for history in self.histories:
            history.reset_parameters()
        for history_out in self.histories_outdate:
            history_out.reset_parameters()

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.convs[-1:]

    # 分cluster更新对应的historical emb.,这里的self是global_model
    def update_hists_cluster(self, node_indexes):
        for hist, hist_out in zip(self.histories, self.histories_outdate):
            x = hist.emb
            hist_out.push(x.index_select(0, torch.tensor(node_indexes)), torch.tensor(node_indexes))

    # 对应federated_GCN_embedding_update_periodic
    def forward(self, x: Tensor, adj_t: SparseTensor, global_model,
                adj_t_partial,
                in_com_train_nei_indexes,
                in_data_nei_indexes,
                local_nodes, in_x_local_node_index, epoch, period):
        # if self.drop_input:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        index = 0
        for conv, hist, hist_out in zip(self.convs[:-1], global_model.histories, global_model.histories_outdate):
            if index == 0 and adj_t_partial != None:  # 第一层GCN的emb.的计算
                h = conv(x, adj_t_partial)
            else:
                h = conv(x, adj_t)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if in_com_train_nei_indexes != None and index <= self.num_layers:  # 最后一层不需要pull
                # 当前layer的cross-client neighbors的indexes,从global_model.histories_outdate获取
                # 对应的indexes为其在整个数据集data中的index，即in_data_nei_indexes
                cross_neig_hist = hist_out.pull(in_data_nei_indexes[index])
                # 更新当前client上的emb.，使用的是cross-client neighbors在训练数据中的index，即in_com_train_nei_indexes
                x[in_com_train_nei_indexes[index], :] = cross_neig_hist
                # 将当前client的local nodes写入global_model.histories中
                # 这里暂时没问题，因为2-layer GCN中hist只有一个
                hist.push(x.index_select(0, in_x_local_node_index), local_nodes)
            index += 1

        h = self.convs[-1](x, adj_t)
        return torch.log_softmax(h, dim=-1)

    # 对应federated_GCN_embedding_update_realtime
    def forward2(self, x: Tensor, adj_t: SparseTensor, global_model,
                 adj_t_partial: Optional = None,
                 in_com_train_nei_indexes: Optional = None,
                 in_data_nei_indexes: Optional = None,
                 local_nodes: Optional = None, in_x_local_node_index: Optional = None, *args):
        # if self.drop_input:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        index = 0
        for conv, hist in zip(self.convs[:-1], global_model.histories):
            if index == 0 and adj_t_partial != None:  # 第一层GCN的emb.的计算
                h = conv(x, adj_t_partial)
            else:
                h = conv(x, adj_t)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_copy = x.clone()
            # print("x_copy.eq(x):",x_copy.eq(x))

            if in_com_train_nei_indexes != None:
                # 当前layer的cross-client neighbors的indexes,从global_model.histories获取
                # 对应的indexes为其在整个数据集data中的index，即in_data_nei_indexes
                cross_neig_hist = hist.pull(in_data_nei_indexes[self.num_layers - 1 - index])
                # 更新当前client上的emb.，使用的是cross-client neighbors在训练数据中的index，即in_com_train_nei_indexes
                x[in_com_train_nei_indexes[self.num_layers - 1 - index], :] = cross_neig_hist
            index += 1
            # print("x_copy.eq(x):", x_copy.eq(x))

        h = self.convs[-1](x, adj_t)
        if local_nodes != None:
            # 将当前client的local nodes写入global_model.histories中
            hist.push(h.index_select(0, in_x_local_node_index), local_nodes)

        return torch.log_softmax(h, dim=-1)

    # 与federated_GCN_embedding配套的
    def forward2(self, x: Tensor, adj_t: SparseTensor,
                 adj_t_partial: Optional = None,
                 indexes_neighbors_layers: Optional = None, *args) -> Tensor:
        # if self.drop_input:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        index = 0
        for conv, hist in zip(self.convs[:-1], self.histories):
            if index == 0 and adj_t_partial != None:  # 第一层GCN的emb.的计算
                h = conv(x, adj_t_partial)
            else:
                h = conv(x, adj_t)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_copy = x.clone()
            # print("x_copy.eq(x):",x_copy.eq(x))

            if indexes_neighbors_layers != None:
                # 当前layer的cross-client neighbors的indexes
                cross_neig_hist = hist.pull(indexes_neighbors_layers[self.num_layers - 1 - index])
                x[indexes_neighbors_layers[self.num_layers - 1 - index], :] = cross_neig_hist
            index += 1
            # print("x_copy.eq(x):", x_copy.eq(x))

            x = self.push_and_pull(hist, x, *args)

        h = self.convs[-1](x, adj_t)

        return torch.log_softmax(h, dim=-1)

    def forward1(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.linear:
        #     x = self.lins[0](x).relu_()
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
            h = conv(x, adj_t)
            # if self.batch_norm:
            #     h = bn(h)
            # if self.residual and h.size(-1) == x.size(-1):
            #     h += x[:h.size(0)]
            x = h.relu_()
            x = self.push_and_pull(hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, adj_t)

        if not self.linear:
            return h

        # if self.batch_norm:  # 因为self.batch_norm = False
        #     h = self.bns[-1](h)
        # if self.residual and h.size(-1) == x.size(-1):  # 因为self.residual = False
        #     h += x[:h.size(0)]
        # h = h.relu_()  # 执行不到这里
        # h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.lins[1](h)

    def push_and_pull(self, history, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""

        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            h = history.pull(n_id[batch_size:])
            return torch.cat([x[:batch_size], h], dim=0)

        else:
            out = self.pool.synchronize_pull()[:n_id.numel() - batch_size]
            self.pool.async_push(x[:batch_size], offset, count, history.emb)
            out = torch.cat([x[:batch_size], out], dim=0)
            self.pool.free_pull()
            return out

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # if self.linear:
            #     x = self.lins[0](x).relu_()
            #     x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t)

        if layer < self.num_layers - 1:
            # if self.batch_norm:  # self.batch_norm = False
            #     h = self.bns[layer](h)
            # if self.residual and h.size(-1) == x.size(-1):  # self.batch_norm = True
            #     h += x[:h.size(0)]
            h = h.relu_()

        # if self.linear:
        #     h = F.dropout(h, p=self.dropout, training=self.training)
        #     h = self.lins[1](h)

        return h


# 跟GCN2的主要区别在于保存旧的cross-client neighbor emb.的不同：GCN3用每个client的histories，而GCN2用global model的histories_outdate
class GCN3(GCN2):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers, num_nodes):
        # super(GCN2, self).__init__(nfeat, nhid, nclass, dropout, NumLayers, num_nodes)
        super().__init__(nfeat, nhid, nclass, dropout, NumLayers, num_nodes)

    # client从global.histories pull最新的cross-client neighbor emb.
    def pull_latest_hists(self, global_model, node_indexes):
        for global_hist, local_hist in zip(global_model.histories, self.histories):
            x = global_hist.emb
            local_hist.push(x.index_select(0, torch.tensor(node_indexes)), torch.tensor(node_indexes))

    # 对应federated_GCN_embedding_update_periodic
    def forward(self, x: Tensor, adj_t: SparseTensor, global_model,
                adj_t_partial,
                in_com_train_nei_indexes,
                in_data_nei_indexes,
                local_nodes, in_x_local_node_index, epoch, period):

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        index = 0
        for conv, local_hist, global_hist in zip(self.convs[:-1], self.histories, global_model.histories):
            if index == 0 and adj_t_partial != None:  # 第一层GCN的emb.的计算
                h = conv(x, adj_t_partial)
            else:
                h = conv(x, adj_t)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if in_com_train_nei_indexes != None and index <= self.num_layers:
                # 当前layer的cross-client neighbors的indexes,从global_model.histories_outdate获取
                # 对应的indexes为其在整个数据集data中的index，即in_data_nei_indexes
                cross_neig_hist = local_hist.pull(in_data_nei_indexes[index])
                # 更新当前client上的emb.，使用的是cross-client neighbors在训练数据中的index，即in_com_train_nei_indexes
                x[in_com_train_nei_indexes[index], :] = cross_neig_hist
                # 将当前client的local nodes写入global_model.histories中
                # 这里暂时没问题，因为2-layer GCN中hist只有一个
                global_hist.push(x.index_select(0, in_x_local_node_index), local_nodes)
            index += 1
            # print("x_copy.eq(x):", x_copy.eq(x))

        h = self.convs[-1](x, adj_t)
        return torch.log_softmax(h, dim=-1)


class GCN4(GCN2):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers, num_nodes):
        # super(GCN2, self).__init__(nfeat, nhid, nclass, dropout, NumLayers, num_nodes)
        super().__init__(nfeat, nhid, nclass, dropout, NumLayers, num_nodes)

    def forward(self, x: Tensor, adj_t: SparseTensor, global_model,
                adj_t_partial,
                in_com_train_nei_indexes,
                in_data_nei_indexes,
                local_nodes, in_x_local_node_index, epoch, period):
        # if self.drop_input:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # list[:-1],作用是返回从start_index = 0到end_index = -1的一串数据，不包含最后一个元素
        index = 0
        for conv, hist, hist_out in zip(self.convs[:-1], global_model.histories, global_model.histories_outdate):
            if index == 0 and adj_t_partial != None:  # 第一层GCN的emb.的计算
                h = conv(x, adj_t_partial)
            else:
                h = conv(x, adj_t)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 最后一层不需要pull
            if in_com_train_nei_indexes != None and index <= self.num_layers and index < 1:
                # 当前layer的cross-client neighbors的indexes,从global_model.histories_outdate获取
                # 对应的indexes为其在整个数据集data中的index，即in_data_nei_indexes
                cross_neig_hist = hist_out.pull(in_data_nei_indexes[index])
                # 更新当前client上的emb.，使用的是cross-client neighbors在训练数据中的index，即in_com_train_nei_indexes
                x[in_com_train_nei_indexes[index], :] = cross_neig_hist
                # 将当前client的local nodes写入global_model.histories中
                # 这里暂时没问题，因为2-layer GCN中hist只有一个
                hist.push(x.index_select(0, in_x_local_node_index), local_nodes)
            index += 1

        h = self.convs[-1](x, adj_t)
        return torch.log_softmax(h, dim=-1)