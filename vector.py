import csv
import networkx as nx
import numpy as np
import torch
from rdkit import Chem


"""
The following code will convert the SMILES format into onehot format
"""


# def atom_features(atom):
#     return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
#                                           ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
#                                            'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
#                                            'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
#                                            'Pt', 'Hg', 'Pb', 'Unknown']) +
#                     one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     [atom.GetIsAromatic()])

def atom_features(atom):
    HYB_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER]
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Sm', 'Tc', 'Gd', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetHybridization(), HYB_list) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    # lambda 定义一个匿名函数
    # map 遍历allowable_set的每个元素，执行lambda函数，返回由函数返回值组成的列表
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    # 读取smile,smiles转换为分子对象，转为2D图
    # print(smile)
    mol = Chem.MolFromSmiles(smile)

    # print(type(mol))
    # 图的顶点数量
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        # 上个函数，独热编码格式
        feature = atom_features(atom)
        # 归一化？？？
        # features.append(feature / sum(feature))
        features.append(feature)

    features = np.array(features)
    # features = features / np.sum(features, 0)
    # features[np.isnan(features)] = 0

    edges = []
    edge_type = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_type.append(bond.GetBondTypeAsDouble())
    # 返回图形的有向表示，
    # 返回值：G –具有相同名称，相同节点且每个边（u，v，数据）由两个有向边（u，v，数据）和（v，u，数据）替换的有向图。
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    if not edge_index:
        edge_index = []
    else:
        edge_index = np.array(edge_index).transpose(1, 0)

    return c_size, features, edge_index, edge_type


def load_drug_smile(file):
    """
    :return: drug_dict {} 键值对为 name: 序号,
             drug_smile [] 所有drug的smile
             # smile_graph {} 键值对为 simle: graph
    """
    reader = csv.reader(open(file))
    # next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[1]
        # 除去重复的name，字典键值对为name-序号
        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    """    
    # 将smile转化为图结构（内部再转化为独热编码）
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    """
    return drug_dict, drug_smile


def convert2graph(drug_smile):
    """
    :param drug_smile: list
    :return: smile_graph {} 键值对为 simle: graph
    """
    # 将smile转化为图结构（内部再转化为独热编码）
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g # g得到的是元组(原子个数，原子特征，边索引，边类型)
    return smile_graph

# 将副作用特征更新为药物集合特征
# 将副作用特征更新为药物集合特征
def update_side_effect_features_loop(data, data_e, x):
    """
    根据频率矩阵更新副作用特征，药物特征通过全局最大池化获得嵌入。
    """
    frequencyMat = data_e.frequencyMat  # 获取频率矩阵

    # t_row_sums = np.sum(frequencyMat, axis=1) 用于判断是否存在全为0的行
    # t_row_zeros_index = np.where(t_row_sums == 0)[0]

    # 确保 frequencyMat 是一个 torch.Tensor
    if not isinstance(frequencyMat, torch.Tensor):
        frequencyMat = torch.tensor(frequencyMat, dtype=torch.float)
    frequencyMat = frequencyMat.to(x.device)

    # row_sums = torch.sum(frequencyMat, axis=1)
    # row_zeros_index = torch.where(row_sums == 0)[0]

    num_side_effects, num_drugs = frequencyMat.shape

    # 1. 获取药物嵌入
    drug_embeddings = x  # 药物的特征矩阵

    # 2. 初始化副作用特征
    embedding_dim = drug_embeddings.size(1)
    side_effect_features = torch.zeros(num_side_effects, embedding_dim)

    # 处理当前批次药物的索引（从 data.index 获取）
    batch_drug_indices = torch.tensor(data.index, dtype=torch.long).to(x.device).flatten() # 确保是 Tensor 类型，并且是 1D
    # print("当前批次药物索引：", batch_drug_indices)


    # 3. 聚合药物特征到副作用
    for side_effect_idx in range(num_side_effects):
        # 获取产生该副作用的药物索引
        drug_indices = torch.where(frequencyMat[side_effect_idx] > 0)[0]

        # 通过 frequencyMat 和 batch_drug_indices，过滤出在当前批次中符合的药物
        valid_drug_indices = torch.isin(batch_drug_indices, drug_indices)
        valid_drug_indices_positions = torch.nonzero(valid_drug_indices).flatten()

        if valid_drug_indices_positions.numel() > 0:  # 如果有药物符合条件
            # 获取符合条件的药物索引
            # 取对应药物嵌入的均值作为副作用特征
            side_effect_features[side_effect_idx] = drug_embeddings[valid_drug_indices_positions].mean(dim=0)
            # print("yes:",side_effect_idx)
        else:
            # 如果没有药物，保留零向量（已初始化为 0，无需重复赋值）
            side_effect_features[side_effect_idx] = torch.zeros(embedding_dim)
            # print("no:",side_effect_idx)

    return side_effect_features

def update_side_effect_features(data, data_e, x, is_train):
    """
    根据频率矩阵更新副作用特征，药物特征通过全局最大池化获得嵌入。
    """
    # if is_train:
    #     print("True")
    # else:
    #     print("no train")
    frequencyMat = data_e.frequencyMat  # 获取频率矩阵

    # 确保 frequencyMat 是一个 torch.Tensor
    if not isinstance(frequencyMat, torch.Tensor):
        frequencyMat = torch.tensor(frequencyMat, dtype=torch.float)
    frequencyMat = frequencyMat.to(x.device)

    # 获取批次中的药物索引
    batch_drug_indices = torch.tensor(data.index, dtype=torch.long).flatten().to(x.device)

    # 构造批次药物的掩码矩阵
    # 每一行对应一个副作用，每一列对应当前批次的药物
    batch_mask = torch.zeros(frequencyMat.size(0), len(batch_drug_indices), device=x.device, dtype=torch.bool)

    # 为掩码矩阵赋值，指示哪些药物属于当前批次并且产生该副作用
    for i, batch_idx in enumerate(batch_drug_indices):
        batch_mask[:, i] = frequencyMat[:, batch_idx] > 0

    # 使用掩码聚合药物嵌入到副作用特征
    # 如果 batch_mask 的形状是 [num_side_effects, batch_size]
    # 则 drug_embeddings 的形状是 [batch_size, embedding_dim]
    # 聚合结果的形状将是 [num_side_effects, embedding_dim]
    mask_sum = batch_mask.sum(dim=1, keepdim=True).clamp(min=1)  # 防止除以 0
    side_effect_features = torch.matmul(batch_mask.float(), x) / mask_sum

    # 创造掩码矩阵(994*1)，1代表副作用特征不全为0，0代表副作用特征全为0
    # 计算布尔掩码，标识哪些行的 x_e_include_drug 不全为零
    # mask = (side_effect_features.abs().sum(dim=1) > 0).unsqueeze(1)  # 在第1维扩展，方便与 x_e 和 x_e_include_drug 的形状匹配
    mask = (batch_mask.float().abs().sum(dim=1) > 0).unsqueeze(1)
    return side_effect_features, mask

def new_update_side_effect_features(data, data_e, x, is_train):
    """
    根据频率矩阵更新副作用特征，药物特征通过动态权重聚合获得嵌入。
    """
    frequencyMat = data_e.frequencyMat
    if not isinstance(frequencyMat, torch.Tensor):
        frequencyMat = torch.tensor(frequencyMat, dtype=torch.float)
    frequencyMat = frequencyMat.to(x.device)

    # 获取药物的权重
    drug_weights = torch.matmul(frequencyMat, x)  # [994, embedding_dim]
    weight_sum = frequencyMat.sum(dim=1, keepdim=True).clamp(min=1)  # 防止除以0
    side_effect_features = drug_weights / weight_sum  # 加权聚合

    # 融合 GCN 提取的特征
    if is_train:
        mask = (frequencyMat.sum(dim=1) > 0).unsqueeze(1)  # 标记是否存在药物贡献
    else:
        mask = torch.ones((frequencyMat.size(0), 1), device=x.device)  # 推理阶段全覆盖

    return side_effect_features, mask



if __name__ == '__main__':
    # drug_dict, drug_smile = load_drug_smile('./data_WS/drug_SMILES.csv')
    # print(drug_dict)
    # smile = drug_smile[0: 10]
    # smile_graph = convert2graph(smile)
    # a = smile_graph[smile[1]]
    # print(a)
    # print(a[0])
    # print(np.asarray(a[1]).shape)
    # b = np.asarray(a[1])
    # print(b[:, 0])
    # print(b[0])
    # smile_graph = convert2graph(['[Cl-].[Cl-].[223Ra++]', 'O.O.O.[OH-].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[Na+].[Na+].[Fe+3].[Fe+3].[Fe+3].[Fe+3].[Fe+3].OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O'])
    # print(type(smile_graph))
    # print(smile_graph['O.O.O.[OH-].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[O--].[Na+].[Na+].[Fe+3].[Fe+3].[Fe+3].[Fe+3].[Fe+3].OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O'])
    # print(type(smile_graph['CS(=O)(=O)OCCCCOS(C)(=O)=O']))
    # print(np.asarray(smile_graph['CS(=O)(=O)OCCCCOS(C)(=O)=O'][1]).shape)

    pass
