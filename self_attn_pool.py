import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import torch_scatter
import torch.optim as optim
from data_prepare import *


def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5,

    Args:
        adjacency: sp.csr_matrix.

    Returns:
        归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
    print(1)
    degree = np.array(adjacency.sum(1))
    print(2)
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    print(3)
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    print(4)
    # 转换为 torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    print(5)
    values = torch.from_numpy(L.data.astype(np.float32))
    print(6)
    tensor_adjacency = torch.sparse.DoubleTensor(indices, values, L.shape)
    print(7)
    return tensor_adjacency


def filter_adjacency(adjacency, mask):
    """根据掩码mask对图结构进行更新
    
    Args:
        adjacency: torch.sparse.FloatTensor, 池化之前的邻接矩阵
        mask: torch.Tensor(dtype=torch.bool), 节点掩码向量
    
    Returns:
        torch.sparse.FloatTensor, 池化之后归一化邻接矩阵
    """
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)


def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


# class DDDataset(object):
#     url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"
#
#     def __init__(self, data_root="data", train_size=0.8):
#         self.data_root = data_root
#         self.maybe_download()
#         sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
#         self.sparse_adjacency = sparse_adjacency.tocsr()
#         self.node_labels = node_labels
#         self.graph_indicator = graph_indicator
#         self.graph_labels = graph_labels
#         self.train_index, self.test_index = self.split_data(train_size)
#         self.train_label = graph_labels[self.train_index]
#         self.test_label = graph_labels[self.test_index]
#
def split_data(graph_indicator, train_size):
    unique_indicator = np.asarray(list(set(graph_indicator)))
    train_index, test_index = train_test_split(unique_indicator,
                                               train_size=train_size,
                                               random_state=1234)
    return train_index, test_index
#
#     def __getitem__(self, index):
#         mask = self.graph_indicator == index
#         node_labels = self.node_labels[mask]
#         graph_indicator = self.graph_indicator[mask]
#         graph_labels = self.graph_labels[index]
#         adjacency = self.sparse_adjacency[mask, :][:, mask]
#         return adjacency, node_labels, graph_indicator, graph_labels
#
#     def __len__(self):
#         return len(self.graph_labels)
#
#     def read_data(self):
#         data_dir = os.path.join(self.data_root, "DD")
#         print("Loading DD_A.txt")
#         adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
#                                        dtype=np.int64, delimiter=',') - 1
#         print("Loading DD_node_labels.txt")
#         node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"),
#                                     dtype=np.int64) - 1
#         print("Loading DD_graph_indicator.txt")
#         graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"),
#                                         dtype=np.int64) - 1
#         print("Loading DD_graph_labels.txt")
#         graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"),
#                                      dtype=np.int64) - 1
#         num_nodes = len(node_labels)
#         sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)),
#                                           (adjacency_list[:, 0], adjacency_list[:, 1])),
#                                          shape=(num_nodes, num_nodes), dtype=np.float32)
#         print("Number of nodes: ", num_nodes)
#         # node_infos = pd.DataFrame(columns=["node_labels", "graph_indicator"])
#         # node_infos["node_labels"] = node_labels
#         # node_infos["graph_indicator"] = graph_indicator
#         return sparse_adjacency, node_labels, graph_indicator, graph_labels
#
#     def maybe_download(self):
#         save_path = os.path.join(self.data_root)
#         if not os.path.exists(save_path):
#             self.download_data(self.url, save_path)
#         if not os.path.exists(os.path.join(self.data_root, "DD")):
#             zipfilename = os.path.join(self.data_root, "DD.zip")
#             with ZipFile(zipfilename, "r") as zipobj:
#                 zipobj.extractall(os.path.join(self.data_root))
#                 print("Extracting data from {}".format(zipfilename))
#
#     @staticmethod
#     def download_data(url, save_path):
#         """数据下载工具，当原始数据不存在时将会进行下载"""
#         print("Downloading data from {}".format(url))
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         data = urllib.request.urlopen(url)
#         filename = "DD.zip"
#         with open(os.path.join(save_path, filename), 'wb') as f:
#             f.write(data.read())
#         return True


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        print(type(self.weight))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        support = torch.mm(input_feature, self.weight)

        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)
    
    def forward(self, adjacency, input_feature, graph_indicator):
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)
        
        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency
    
    
def top_rank(attention_score, graph_indicator, keep_ratio):
    """基于给定的attention_score, 对每个图进行pooling操作.
    为了直观体现pooling过程，我们将每个图单独进行池化，最后再将它们级联起来进行下一步计算
    
    Arguments:
    ----------
        attention_score：torch.Tensor
            使用GCN计算出的注意力分数，Z = GCN(A, X)
        graph_indicator：torch.Tensor
            指示每个节点属于哪个图
        keep_ratio: float
            要保留的节点比例，保留的节点数量为int(N * keep_ratio)
    """
    # TODO: 确认是否是有序的, 必须是有序的
    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score = attention_score[graph_indicator == graph_id]
        graph_node_num = len(graph_attn_score)
        graph_mask = attention_score.new_zeros((graph_node_num,),
                                                dtype=torch.bool)
        keep_graph_node_num = int(keep_ratio * graph_node_num)
        _, sorted_index = graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[:keep_graph_node_num]] = True
        mask = torch.cat((mask, graph_mask))
    
    return mask


class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构A
        
        Args:
        ----
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐藏层单元数
            num_classes: 分类类别数 (default: 2)
        """
        super(ModelA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3, 0.5)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator):
        input_feature = input_feature.double()
        adjacency = adjacency.double()
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))
        
        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,
                                                               graph_indicator)
        
        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1)
        
        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        
        return logits


class ModelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构
        
        Arguments:
        ----------
            input_dim {int} -- 输入特征的维度
            hidden_dim {int} -- 隐藏层单元数
        
        Keyword Arguments:
        ----------
            num_classes {int} -- 分类类别数 (default: {2})
        """
        super(ModelB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, num_classes))
    
    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = \
            self.pool1(adjacency, gcn1, graph_indicator)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)],
            dim=1)
        
        gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],
            dim=1)
        
        readout = global_pool1 + global_pool2 + global_pool3
        
        logits = self.mlp(readout)
        return logits
##loda data
data,labels = load_data("Cirrhosis")
data = data.transpose()
sums = data.sum(axis=1)
data = data.divide(sums, axis=0)
labels, label_set = pd.factorize(labels)
features_raw = list(data.columns.values)
features,features_level,map_values, map_edges = generate_map(data,features_raw)

# 模型输入数据准备

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
node_labels,edges_list,graph_indicator = prepare_GNN_data(map_values,map_edges)
train_index, test_index = split_data(graph_indicator,0.9)
print(np.array(edges_list[0]).shape[0])
print(np.array(edges_list[0]).shape)
print(np.ones(np.array(edges_list[0]).reshape(-1,1).shape[0]).shape)
adjacency = sp.coo_matrix((np.ones(np.array(edges_list[0]).reshape(-1,1).shape[0]),(np.array(edges_list[0]),np.array(edges_list[1]))),shape=(len(node_labels), len(node_labels)),dtype=np.float32)

print("adjacency:"+str(adjacency.shape))

normalize_adjacency = normalization(adjacency).to(DEVICE)

print("normalize_adjacency:"+str(normalize_adjacency.shape))
node_features = torch.from_numpy(node_labels.reshape(-1,1)).to(DEVICE)
node_features = node_features.float()
print("node_features:"+str(node_features.shape))
# node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float()
graph_indicator = torch.from_numpy(graph_indicator).to(DEVICE)
print("graph_indicator:"+str(graph_indicator.shape))

graph_labels = torch.from_numpy(labels).to(DEVICE)
print("graph_labels:"+str(graph_labels.shape))

train_label = labels[train_index]
test_label = labels[test_index]
train_index = torch.from_numpy(train_index).to(DEVICE)
test_index = torch.from_numpy(test_index).to(DEVICE)
train_label = torch.from_numpy(train_label).to(DEVICE)
test_label = torch.from_numpy(test_label).to(DEVICE)

# 超参数设置
print(node_features.size)
INPUT_DIM = node_features.size(1)
print(INPUT_DIM)
NUM_CLASSES = 2
EPOCHS = 200    # @param {type: "integer"}
HIDDEN_DIM =    32# @param {type: "integer"}
LEARNING_RATE = 0.01 # @param
WEIGHT_DECAY = 0.0001 # @param

# 模型初始化
model_g = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model_h = ModelB(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)

model = model_g #@param ['model_g', 'model_h'] {type: 'raw'}

print("Device:", DEVICE)
print(model)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print("######### start training ##########")
model.train()
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency, node_features, graph_indicator)
    loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
    optimizer.zero_grad()
    loss.backward()  # 反向传播计算参数的梯度
    optimizer.step()  # 使用优化方法进行梯度更新
    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(
        epoch, loss.item(), train_acc.item()))
print("######### end training ##########")

model.eval()
with torch.no_grad():
    logits = model(normalize_adjacency, node_features, graph_indicator)
    test_logits = logits[test_index]
    test_acc = torch.eq(
        test_logits.max(1)[1], test_label
    ).float().mean()

print(test_acc.item())