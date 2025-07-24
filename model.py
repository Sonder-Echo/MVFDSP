import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class MVFDSP(torch.nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, output_dim_e=64, dropout=0.2, frequency_mat = None, device=None,heads=10):
        super(MVFDSP, self).__init__()

        # 预处理得到的药物特征进行线向变换
        self.fc_p1 = nn.Linear(128, input_dim)

        # 定义了两个全连接层，用于药物的特征转换
        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.fc_2 = nn.Linear(output_dim, output_dim)

        # 用于对药物特征进行进一步处理
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)

        # 副作用特征的图卷积层 (GAT)
        # GAT层（副作用）
        self.gcn1 = GATConv(input_dim_e, 128, heads=heads)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn3 = GATConv(output_dim * heads, output_dim)
        # 用于对卷积后的副作用特征进行进一步处理
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()

        # 定义多个 LayerNorm 层，用于不同维度的特征归一化
        self.norm1 = nn.LayerNorm([input_dim])
        self.norm2 = nn.LayerNorm([input_dim_e])
        self.norm3 = nn.LayerNorm([output_dim])
        self.norm4 = nn.LayerNorm([output_dim])

        # 对副作用特征进行图卷积后的归一化层
        self.norm_e_1 = nn.LayerNorm([input_dim_e])
        self.norm_e_2 = nn.LayerNorm([1280])
        self.norm_e_3 = nn.LayerNorm([output_dim * heads])

        # 将weight定义为可学习参数
        self.weight = nn.Parameter(torch.tensor(0.5))  # 推荐初始值0.5，可调整

        # 加载药物特征，detach避免计算图积累
        self.drug_feature_1d = torch.load("data_processed/drug_features_1d.pt", map_location=device).detach()
        self.drug_feature_2d = torch.load("data_processed/drug_features_2d.pt", map_location=device).detach()

    def forward(self, data, data_e, DF=False, not_FC = True):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 药物特征及其图结构
        x_e, edge_index_e = data_e.x, data_e.edge_index  # 副作用特征及其图结构

        # 限制weight在0~1之间
        weight = torch.sigmoid(self.weight)
        # 通过weight参数自适应融合药物特征
        drug_feature = weight * self.drug_feature_1d + (1 - weight) * self.drug_feature_2d

        x = torch.stack([drug_feature[idx].squeeze(0) for idx in data.index])
        x = self.relu(self.fc_p1(x))  # 将药物特征维度从128转化为109

        # 首先进行全连接层的特征转换
        x = self.norm1(x)  # 药物特征归一化
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)

        # 副作用图卷积处理部分
        x_e = self.norm_e_1(x_e)  # 副作用特征归一化
        x_e = self.relu(self.gcn1(x_e, edge_index_e))  # 第一层图卷积
        x_e = self.norm_e_2(x_e)  # 归一化
        x_e = self.relu(self.gcn2(x_e, edge_index_e))  # 第二层图卷积
        x_e = self.norm_e_3(x_e)  # 归一化
        x_e = self.gcn3(x_e, edge_index_e)  # 第三层图卷积
        x_e = self.fc_g4(x_e)  # 开这个就要注释掉下面

        # 计算得分
        xc = torch.matmul(x, x_e.T)

        return xc, x, x_e  # 返回药物-副作用相互作用得分和处理后的药物、副作用特征
