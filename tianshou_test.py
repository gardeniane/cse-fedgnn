import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split


# Define the SACDNet for client selection
class SACDNet(nn.Module):
    def __init__(self, input_dim, num_clients, num_selected):
        super(SACDNet, self).__init__()
        self.num_selected = num_selected

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_clients)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        client_probs = self.softmax(self.fc2(x))
        return client_probs


# Define the GCN model
class GCNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCNNet, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Initialize the Cora dataset
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# Split the dataset into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data.x, data.y, test_size=0.2, stratify=data.y, random_state=123)

# Randomly assign data to clients
num_clients = 20
client_data = []
for _ in range(num_clients):
    client_train_data, _, client_train_labels, _ = train_test_split(
        train_data, train_labels, test_size=0.2, stratify=train_labels)
    client_data.append((client_train_data, client_train_labels))

# Define the client selection model
num_selected_clients = 5
client_selection_net = SACDNet(train_data.size(1), num_clients, num_selected_clients)

# Define the GCN model
input_dim = dataset.num_features
hidden_dim = 16
num_classes = dataset.num_classes
gcn_net = GCNNet(input_dim, hidden_dim, num_classes)

# Define the optimizer
optimizer = optim.Adam(gcn_net.parameters(), lr=0.01)

# Training loop
num_epochs = 10
batch_size = 16

for epoch in range(num_epochs):
    # Select clients for training using SACD
    client_probs = client_selection_net(train_data)
    selected_clients = torch.multinomial(client_probs, num_selected_clients, replacement=False).tolist()

    for client_id in selected_clients:

        selected_client_data = [client_data[i] for i in client_id]
        client_train_data = torch.stack([selected_client_data[i][0] for i in range(len(selected_client_data))])
        client_train_labels = torch.stack([selected_client_data[i][1] for i in range(len(selected_client_data))])
        client_loader = DataLoader(client_train_data, batch_size=batch_size, shuffle=True)

        for batch in client_loader:
            x, edge_index = batch
            optimizer.zero_grad()
            client_output = gcn_net(x, edge_index)  # Fixed line
            loss = nn.functional.cross_entropy(client_output, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluate on the test set
    with torch.no_grad():
        test_output = gcn_net(test_data, data.edge_index)
        test_loss = nn.functional.cross_entropy(test_output, test_labels)
        test_accuracy = torch.sum(test_output.argmax(dim=1) == test_labels).item() / len(test_labels)

    print(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
