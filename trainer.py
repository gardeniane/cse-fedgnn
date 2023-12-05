import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler


# Training settings
lr = 0.01
# momentum = 0.9
weight_decay = 5e-4

# read data
dataset_name = "Citeseer"
dataset = Planetoid(root="data/", name=dataset_name)
data = dataset[0]  # Get the first graph object.

# sample a mini-batch
subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=8)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def inference(self, x_all):  # 第二种test()，其中node的embedding利用subgraph_loader生成
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def get_optimizer(model):
    # model.parameters(),即所谓的model，tensor
    # 该语句返回值是一个torch.optim对象，主要内容为params：model.parameters()，和学习率等参数（保存为一个dict）
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# 返回一个mini-batch的subgraph
def get_trainloader(trainset, batch_size):
    # 从给定的数据集trainset（两个向量，x和y，分别保存数据向量和数据label）中随机选择batch_size个数据
    # 返回值是一个DataLoader对象
    return NeighborSampler(trainset.edge_index, sizes=[-1], batch_size=batch_size, shuffle=False, num_workers=8)


def train(model, optimizer, train_loader):
    model.train()  # 前面好像不需要model.to(device)这一步
    criterion = torch.nn.CrossEntropyLoss()

    for sub_data in train_loader:  # Iterate over each mini-batch.
        out = model(sub_data.x, sub_data.edge_index)  # Perform a single forward pass.
        # Compute the loss solely based on the training nodes.
        loss = criterion(out[sub_data.train_mask], sub_data.y[sub_data.train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
    return accs


# if __name__ == "__main__":
#     model = GCN(hidden_channels=16)
#     optimizer = get_optimizer(model)
#     train_loader = get_trainloader(data, batch_size=1024)
#     for epoch in range(1, 51):
#         loss = train(model, optimizer, train_loader)
#         train_acc, val_acc, test_acc = test(model)
#         print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')