import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True):
        super(CustomGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_weight, self_loop_weights=None):
        # Add self loop
        if self_loop_weights is not None:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0),
                                                     fill_value=self_loop_weights)

        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j is a characteristic of neighboring nodes
        # edge_weight  is the corresponding edge weight
        return edge_weight.view(-1, 1) * x_j


class AdditiveAttention(nn.Module):
    def __init__(self, num_features):
        super(AdditiveAttention, self).__init__()
        self.ws = nn.Linear(num_features, 1, bias=False)

    def forward(self, queries, keys, values):
        scores = self.ws(torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1)))
        attention_weights = F.softmax(scores, dim=-1).squeeze(-1)
        new_values = attention_weights.bmm(values)
        return new_values


class QValueNetMultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_heads, num_hiddens):
        super(QValueNetMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.wq = nn.Linear(query_size, num_hiddens, bias=False)
        self.wk = nn.Linear(key_size, num_hiddens, bias=False)
        self.wv = nn.Linear(value_size, num_hiddens, bias=False)
        self.attention = AdditiveAttention(self.num_hiddens // self.num_heads)
        self.wo = nn.Linear(num_hiddens, value_size, bias=False)

    def forward(self, queries, keys, values):
        queries, keys, values = self.wq(queries), self.wk(keys), self.wv(values)

        queries = queries.view(queries.shape[0], queries.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        queries = queries.reshape(-1, queries.shape[2], queries.shape[3])

        keys = keys.view(keys.shape[0], keys.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        keys = keys.reshape(-1, keys.shape[2], keys.shape[3])

        values = values.view(values.shape[0], values.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        values = values.reshape(-1, values.shape[2], values.shape[3])

        out = self.attention(queries, keys, values)
        out = out.view(-1, self.num_heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.wo(out)
        return out


class NodeEncoder(nn.Module):
    def __init__(self, num_features, T=3):
        super(NodeEncoder, self).__init__()
        self.convs1 = nn.Sequential()
        self.convs2 = nn.Sequential()
        for i in range(T):
            self.convs1.add_module(f'conv_1_{i}', CustomGCNConv(num_features, num_features))
            self.convs2.add_module(f'conv_2_{i}', CustomGCNConv(num_features, num_features))
        self.w1 = nn.Linear(num_features, num_features, bias=False)
        self.w2 = nn.Linear(num_features, num_features, bias=False)
        self.fc1 = nn.Linear(2 * num_features, 2 * num_features)
        self.fc2 = nn.Linear(2 * num_features, 1)

    def forward(self, x, edge_index, edge_weight):
        x1 = x[:, x.shape[1] // 2:]
        x2 = x[:, :x.shape[1] // 2]
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            x1 = F.leaky_relu(conv1(x1, edge_index[[1, 0]], edge_weight, None if i == 0 else 1.0), 0.2)
            x2 = F.leaky_relu(conv1(x2, edge_index, edge_weight, None if i == 0 else 1.0), 0.2)
        x = self.fc1(torch.cat([self.w1(x1), self.w2(x2)], dim=-1))
        y = x
        x = F.leaky_relu(self.fc2(x), 0.2).view(-1)
        return x, y


class QValueNetBlock(nn.Module):
    def __init__(self, num_features, bias=False):
        super(QValueNetBlock, self).__init__()
        self.conv = CustomGCNConv(num_features, num_features)
        self.alpha0 = nn.Linear(1, 1, bias=bias)
        self.alpha1 = nn.Linear(num_features, num_features)

    def forward(self, x, edge_index, edge_weights, states, self_weights=None):
        x = F.leaky_relu(self.alpha1(x + self.alpha0(states.view(-1, 1))), 0.2)
        x = self.conv(x, edge_index, edge_weights, self_weights)
        return x


class QValueNet(nn.Module):
    def __init__(self, num_features, T=3, dropout=0.1):
        super(QValueNet, self).__init__()
        self.T = T

        self.encoder = NodeEncoder(num_features, T=T)

        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        for i in range(T):
            self.blocks1.add_module(f'block{i}', QValueNetBlock(num_features, bias=False))
            self.blocks2.add_module(f'block{i}', QValueNetBlock(num_features, bias=False))

        self.attention1 = QValueNetMultiHeadAttention(num_features, num_features, num_features, 4, num_features)
        self.attention2 = QValueNetMultiHeadAttention(num_features, num_features, num_features, 4, num_features)

        self.fc1 = nn.Linear(num_features, 1)
        self.fc2 = nn.Linear(num_features, 1)

        self.beta0 = nn.Linear(num_features, num_features, bias=False)
        self.beta1 = nn.Linear(num_features, num_features, bias=False)
        self.beta2 = nn.Linear(2 * num_features, num_features)

        self.gamma0 = nn.Linear(num_features, num_features, bias=False)
        self.gamma1 = nn.Linear(num_features, num_features, bias=False)
        self.gamma2 = nn.Linear(num_features, num_features, bias=False)
        self.gamma3 = nn.Linear(3 * num_features, 3 * num_features // 2, bias=False)
        self.gamma4 = nn.Linear(3 * num_features // 2, 1)

    def forward(self, x, edge_index, edge_weight, batch, states):
        x = self.encoder(x, edge_index, edge_weight)[1]
        x1 = x[:, : x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]

        x1_list = []
        x2_list = []
        for block1, block2 in zip(self.blocks1, self.blocks2):
            x1 = F.leaky_relu(block1(x1, edge_index[[1, 0]], edge_weight, states), 0.2)
            x2 = F.leaky_relu(block2(x2, edge_index, edge_weight, states), 0.2)
            x1_list.append(x1)
            x2_list.append(x2)

        # attention
        # Combine the node information obtained from each step of aggregation, use multi head attention, and calculate the information for each step of attention aggregation based on the results obtained from multi head attention
        x1 = torch.cat(x1_list, dim=-1).view(-1, self.T, x1.shape[-1])
        x2 = torch.cat(x2_list, dim=-1).view(-1, self.T, x2.shape[-1])
        x1_a = self.attention1(x1, x1, x1)
        x2_a = self.attention2(x2, x2, x2)
        x1_aw = F.softmax(self.fc1(x1_a).permute(0, 2, 1), dim=-1)
        x2_aw = F.softmax(self.fc2(x2_a).permute(0, 2, 1), dim=-1)
        x1 = x1_aw.bmm(x1_a).squeeze(dim=1) + x1.sum(dim=1)
        x2 = x2_aw.bmm(x2_a).squeeze(dim=1) + x2.sum(dim=1)

        # Splicing x1 and x2 to obtain the final vector representation of the node
        x = F.leaky_relu(self.beta2(torch.cat([self.beta0(x1), self.beta1(x2)], dim=-1)), 0.2)

        # Calculate q
        selected_nodes = states == 1
        batch_num = torch_scatter.scatter_add(torch.ones(batch.shape[0], dtype=torch.long, device=batch.device), batch)
        x_s = torch_scatter.scatter_add(x[selected_nodes], batch[selected_nodes], dim=0,
                                        dim_size=batch[-1].item() + 1).repeat_interleave(batch_num, dim=0)
        x_sum = torch_scatter.scatter_add(x, batch, dim=0, dim_size=batch[-1].item() + 1).repeat_interleave(batch_num,
                                                                                                            dim=0)
        x = torch.cat([self.gamma0(x), self.gamma1(x_s), self.gamma2(x_sum)], dim=-1)
        x = F.leaky_relu(self.gamma3(x), 0.2)
        q = self.gamma4(x).view(-1)
        return q


class StudentQValueNet(nn.Module):
    def __init__(self, num_features, T=3, dropout=0.1):
        super(StudentQValueNet, self).__init__()
        self.T = T

        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        for i in range(T):
            self.blocks1.add_module(f'block{i}', QValueNetBlock(num_features, bias=False))
            self.blocks2.add_module(f'block{i}', QValueNetBlock(num_features, bias=False))

        self.beta0 = nn.Linear(num_features, num_features, bias=False)
        self.beta1 = nn.Linear(num_features, num_features, bias=False)
        self.beta2 = nn.Linear(2 * num_features, num_features)

        self.gamma0 = nn.Linear(num_features, num_features, bias=False)
        self.gamma1 = nn.Linear(num_features, num_features, bias=False)
        self.gamma2 = nn.Linear(num_features, num_features, bias=False)
        self.gamma3 = nn.Linear(3 * num_features, 3 * num_features // 2, bias=False)
        self.gamma4 = nn.Linear(3 * num_features // 2, 1)

    def forward(self, x, edge_index, edge_weight, batch, states):
        x1 = x[:, : x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]

        x1_sum = 0
        x2_sum = 0
        for block1, block2 in zip(self.blocks1, self.blocks2):
            x1 = F.leaky_relu(block1(x1, edge_index[[1, 0]], edge_weight, states), 0.2)
            x2 = F.leaky_relu(block2(x2, edge_index, edge_weight, states), 0.2)
            x1_sum += x1
            x2_sum += x2

        x1 = x1_sum
        x2 = x2_sum

        x = F.leaky_relu(self.beta2(torch.cat([self.beta0(x1), self.beta1(x2)], dim=-1)), 0.2)

        selected_nodes = states == 1
        batch_num = torch_scatter.scatter_add(torch.ones(batch.shape[0], dtype=torch.long, device=batch.device), batch)
        x_s = torch_scatter.scatter_add(x[selected_nodes], batch[selected_nodes], dim=0,
                                        dim_size=batch[-1].item() + 1).repeat_interleave(batch_num, dim=0)
        x_sum = torch_scatter.scatter_add(x, batch, dim=0, dim_size=batch[-1].item() + 1).repeat_interleave(batch_num,
                                                                                                            dim=0)
        x = torch.cat([self.gamma0(x), self.gamma1(x_s), self.gamma2(x_sum)], dim=-1)
        x = F.leaky_relu(self.gamma3(x), 0.2)
        q = self.gamma4(x).view(-1)
        return q
