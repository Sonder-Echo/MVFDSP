import os

import argparse
import csv
import datetime
import shutil

import networkx as nx
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader

from model import *
from vector import load_drug_smile, convert2graph
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
seed = 2
input_dim = 109

import numpy as np
torch.manual_seed(seed)

# loss function
def loss_fun(output, label, lam, eps):
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)
    loss = torch.sum((output[x1] - label[x1]) ** 2) + lam * torch.sum((output[x0] - eps) ** 2)
    loss = loss / label.size(0)
    return loss

def generateMat(k=10):
    """
    生成掩码矩阵
    """
    # Clear the original data
    filenames = os.listdir('data_processed/processed')
    print(filenames)
    for s in filenames:
        os.remove('data_processed/processed/' + s)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # Generate mask Mat
    index_pair = np.where(raw != 0)
    index_arr = np.arange(0, index_pair[0].shape[0], 1)
    np.random.shuffle(index_arr)
    x = []
    n = math.ceil(index_pair[0].shape[0] / k)
    for i in range(k):
        if i == k - 1:
            x.append(index_arr[0:].tolist())
        else:
            x.append(index_arr[0:n].tolist())
            index_arr = index_arr[n:]

    dic = {}
    for i in range(k):
        mask = np.ones(raw.shape)
        mask[index_pair[0][x[i]], index_pair[1][x[i]]] = 0
        dic['mask' + str(i)] = mask
    scipy.io.savemat(mask_mat_file, dic)

def split_data(tenfold=False):
    """
    读取 data/mask_mat.mat，根据原始频率矩阵生成10份被mask的频率矩阵并yield
    :return:
    """
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    mask_mat = scipy.io.loadmat(mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)

    simle_graph = convert2graph(drug_smile)
    dataset = 'drug_sideEffect'

    for i in range(10):
        mask = mask_mat['mask' + str(i)]
        frequencyMat = raw * mask

        _ = myDataset(root='data_processed', dataset=dataset + '_data' + str(i), drug_simles=drug_smile,
                      frequencyMat=frequencyMat, simle_graph=simle_graph)
        yield i, frequencyMat, mask

        if not tenfold and i == 0:
            break


def train(model, device, train_loader, optimizer, lamb, epoch, log_interval, sideEffectsGraph, raw, id, DF, not_FC, eps):
    """
    :param model:
    :param device:
    :param train_loader: 数据加载器
    :param optimizer: 优化器
    :param epoch: 训练数
    :param log_interval: 记录间隔
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :param id: 第id次训练(第id折）
    :return: 本次训练的平均损失
    """
    model.train()

    avg_loss = []

    for batch_idx, data in enumerate(train_loader):
        # 查找被mask的数据
        label = data.y
        sideEffectsGraph = sideEffectsGraph.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        out, x, x_e = model(data, sideEffectsGraph, DF, not_FC)

        pred = out.to(device)
        train_label = torch.FloatTensor(label)
        loss = loss_fun(pred.flatten(), train_label.flatten().to(device), lamb, eps)
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

    return sum(avg_loss) / len(avg_loss)

def predict(model, device, loader, sideEffectsGraph, raw, DF, not_FC):
    """
    model prediction function
    :param model: 模型
    :param device: cuda
    :param loader: 数据加载器
    :param sideEffectsGraph: 副作用图信息，
    :param raw: 原始数据
    :return: 所有的被mask的原始值，所有的被mask的预测值，都是1维 # 这里只是被掩码掉的位置预测值和真实值
    """
    # 声明为张量
    total_preds = []
    total_reals = []
    model.eval()
    torch.cuda.manual_seed(seed)

    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            indices = [x[0] for x in data.index]

            label = data.y
            raw_label = torch.FloatTensor(raw[indices])  # 同维度
            index_pair = torch.where(raw_label != label)  # 被掩码的位置

            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)

            # 索引所有掩码位置的数据
            pred_batch = output.cpu()[index_pair].numpy().flatten()
            real_batch = raw_label[index_pair].numpy().flatten()

            total_preds.append(pred_batch)
            total_reals.append(real_batch)

    total_preds = np.concatenate(total_preds)
    total_reals = np.concatenate(total_reals)

    return total_reals, total_preds


def getAllResultMatrix(model, device, loader, sideEffectsGraph, mask, result_folder, DF, not_FC, id):
    """
    保存预测结果
    """
    # 声明为张量
    pred = torch.Tensor()
    model.eval()
    torch.cuda.manual_seed(seed)
    # 加载数据
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = torch.cat((pred, output.cpu()), 0)
        # 保存此次预测的所有结果 750*994
        pred = pred.numpy()
        pred_result = pred

        pred_result = pd.DataFrame(pred_result)
        pred_result.to_csv(result_folder + f'/pred_result_fold{id}.csv', header=False, index=False, float_format='%.4f')

def evaluate(model, device, loader, sideEffectsGraph, mask, raw, DF, not_FC, best_auc, fold_id, result_folder):
    """
    评估函数，用于计算模型在测试集上的各种性能指标
    """
    model.eval()
    torch.cuda.manual_seed(seed)
    total_preds = []


    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for data in loader:
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            total_preds.append(output.cpu())

    total_preds = torch.cat(total_preds, 0).numpy()

    # 获取所有正负样本
    pos_ind = np.where(mask == 0)
    pos_scores = total_preds[pos_ind]
    pos_labels = np.ones(len(pos_scores))

    neg_all_ind = np.where(raw == 0)
    # 采样负样本
    if len(neg_all_ind[0]) > len(pos_scores):
        selected_indices = np.random.choice(len(neg_all_ind[0]), len(pos_scores), replace=False)
        neg_ind = (neg_all_ind[0][selected_indices], neg_all_ind[1][selected_indices])
    else:
        neg_ind = neg_all_ind

    neg_scores = total_preds[neg_ind]
    neg_labels = np.zeros(len(neg_scores))

    y = np.hstack((pos_scores, neg_scores))
    y_true = np.hstack((pos_labels, neg_labels))

    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)

    # 保存auc的label和score以及正负样本标签
    auc_result_folder = result_folder + f'/auc_result/fold{fold_id}/'
    pred_result_folder = result_folder + '/pred_result/'

    if auc_all > best_auc:
        df_auc = pd.DataFrame({'label': y_true, 'score': y})
        df_auc.to_csv(auc_result_folder + "auc_label_score.csv", index=False)

        df_pos = pd.DataFrame({'drug_idx': pos_ind[0], 'se_idx': pos_ind[1], 'score': pos_scores})
        df_pos.to_csv(auc_result_folder + "pos_index_score.csv", index=False)

        df_neg = pd.DataFrame({'drug_idx': neg_ind[0], 'se_idx': neg_ind[1], 'score': neg_scores})
        df_neg.to_csv(auc_result_folder + "neg_index_score.csv", index=False)

        pred_result = pd.DataFrame(total_preds)
        pred_result.to_csv(pred_result_folder + f'pred_result_fold{fold_id}.csv', header=False, index=False,
                           float_format='%.4f')


    # others
    Tr_neg = {}
    Te = {}
    train_data = raw * mask
    Te_pairs = np.where(mask == 0)
    Tr_neg_pairs = np.where(train_data == 0)
    Te_pairs = np.array(Te_pairs).transpose()
    Tr_neg_pairs = np.array(Tr_neg_pairs).transpose()
    for te_pair in Te_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)

    for te_pair in Tr_neg_pairs:
        drug_id = te_pair[0]
        SE_id = te_pair[1]
        if drug_id not in Tr_neg:
            Tr_neg[drug_id] = [SE_id]
        else:
            Tr_neg[drug_id].append(SE_id)

    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Tr_neg, Te, positions)

    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15

def main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name, frequencyMat,
         id, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca):
    print('\n=======================================================================================')
    print('\n第 {} 次训练：\n'.format(id))
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('Batch size: ', train_batch)
    print('Lambda: ', lamb)
    print('weight_decay: ', weight_decay)
    print('KNN: ', knn)
    print('metric: ', metric)
    print('tenfold: ', tenfold)
    print('DF: ', DF)
    print('not_FC: ', not_FC)
    print('output_dim: ', output_dim)
    print('Eps: ', eps)
    print('PCA: ', pca)


    model_st = modeling.__name__
    dataset = 'drug_sideEffect'
    train_losses = []
    print('\nrunning on ', model_st + '_' + dataset)
    processed_raw = raw_file

    if not os.path.isfile(processed_raw):
        print('Missing FrequencyMat, exit!!!')
        exit(1)

    # 生成副作用的graph信息
    frequencyMat = frequencyMat.T
    if pca:
        pca_ = PCA(n_components=256)
        similarity_pca = pca_.fit_transform(frequencyMat)
        print('PCA 信息保留比例： ')
        print(sum(pca_.explained_variance_ratio_))
        A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
    else:
        A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
    G = nx.from_numpy_array(A.todense())
    edges = []
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])

    edges = np.array(edges).T
    edges = torch.tensor(edges, dtype=torch.long)

    # load  side_effect_label mat
    node_label = scipy.io.loadmat(side_effect_label)['node_label']
    feat = torch.tensor(node_label, dtype=torch.float)

    sideEffectsGraph = Data(x=feat, edge_index=edges, frequencyMat=frequencyMat)

    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_processed', dataset='drug_sideEffect_data' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    # masked datas
    test_loader = DataLoader(train_data, batch_size=train_batch, shuffle=False)

    print('CPU/GPU: ', torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = modeling(input_dim=input_dim, output_dim=output_dim, frequency_mat=frequencyMat, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    result_log = result_folder + '/' + model_st + '_result.csv'
    loss_fig_name = str(id) + model_st + '_loss'

    # 记录最佳指标
    best_auc = 0
    best_auc_epoch = 0
    best_scc = 0
    best_metrics = {}
    lowest_loss = float("inf")
    best_epoch = 0
    early_stop_epoch = 0

    scc_result_folder = result_folder + f'/scc_result/fold{id}/'
    epoch_metrics_file = result_folder + f'/epoch_metrics/metrics_epoch_fold{id}.csv'
    modelsFolder = result_folder + '/model_save/'

    for epoch in range(num_epoch):
        all_start_time = datetime.datetime.now()

        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps)
        train_losses.append(train_loss)


        test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                          sideEffectsGraph=sideEffectsGraph, raw=raw, DF=DF, not_FC=not_FC)

        ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                    spearman(test_labels, test_preds), MAE(test_labels, test_preds)]

        test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]


        # 计算 test_auc 和 test_aupr
        test_auc, test_aupr, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(
            model=model,
            device=device,
            loader=test_loader,
            sideEffectsGraph=sideEffectsGraph,
            mask=mask,
            raw=raw,
            DF=DF,
            not_FC=not_FC,
            best_auc=best_auc,
            fold_id=id,
            result_folder=result_folder,
        )

        # 计算本 epoch 耗时
        end_time = datetime.datetime.now()
        epoch_duration = (end_time - all_start_time).total_seconds()


        # 保存每一折的指标
        epoch_metrics = {
            "epoch": epoch + 1,
            "pearson": test_pearsons, "rMSE": test_rMSE, "spearman": test_spearman, "MAE": test_MAE,
            "test_auc": test_auc, "test_aupr": test_aupr, "MAP": map_value, "nDCG": ndcg,
            "P1": p1, "P5": p5, "P10": p10, "P15": p15,
            "R1": r1, "R5": r5, "R10": r10, "R15": r15
        }
        epoch_metrics = {key: round(value, 4) for key, value in epoch_metrics.items()}

        # 如果是第一个 epoch，写入表头
        if epoch == 0:
            with open(epoch_metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
                writer.writeheader()
                writer.writerow(epoch_metrics)
        else:
            with open(epoch_metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
                writer.writerow(epoch_metrics)


        if epoch % 500 == 0:
            print(f"Fold {id}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, "
                  f" test_auc = {test_auc:.4f}, test_aupr = {test_aupr:.4f},\n"
                  f"time = {epoch_duration:.2f}s")

        # 更新最低 loss
        if train_loss < lowest_loss:
            lowest_loss = train_loss
            best_epoch = epoch + 1

        # 更新最佳 auc 及其对应的评估指标
        if test_auc > best_auc:
            early_stop_epoch = 0
            best_auc = test_auc
            best_auc_epoch = epoch + 1

            # 记录当前最佳 auc 对应的所有指标
            best_metrics = {
                "pearson": test_pearsons, "rMSE": test_rMSE, "spearman": test_spearman, "MAE": test_MAE,
                "test_auc": test_auc, "test_aupr": test_aupr, "MAP": map_value, "nDCG": ndcg,
                "P1": p1, "P5": p5, "P10": p10, "P15": p15,
                "R1": r1, "R5": r5, "R10": r10, "R15": r15
            }

            # 在 AUC 最佳时，输出所有评估指标
            best_metrics = {key: round(value, 4) for key, value in best_metrics.items()}
            # print(f"Best AUC Metrics: {best_metrics}")
            print(f"best_auc = {best_auc:.4f} with best_auc_epoch = {best_auc_epoch}")

            # # 写入预测值，同一fold会覆盖
            # getAllResultMatrix(model=model, device=device, loader=test_loader, sideEffectsGraph=sideEffectsGraph,
            #                    mask=mask, result_folder=result_folder, DF=DF, not_FC=not_FC, id=id)

            # 写入预测指标，第一次写入需要重启一行，后面需要重写最后一行
            result = list(best_metrics.values())
            result.insert(0, f'fold{id}')

            if epoch == 0:
                with open(result_log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
            else:
                # 生成要写入的结果行
                new_result_line = ",".join(map(str, result)) + f"\r\n"
                with open(result_log, 'r+', newline='') as f:
                    lines = f.readlines()
                    # 覆盖当前 CSV 文件最后一行（即当前 fold 的数据）
                    lines[-1] = new_result_line
                    f.seek(0)  # 回到文件开头
                    f.writelines(lines)  # 重新写入所有行
                    f.truncate()  # 清除多余的旧内容

            if save_model:
                torch.save(model, modelsFolder + f'model_fold{id}.pth')

        early_stop_epoch += 1
        if early_stop_epoch > 2000:
            break

    # 在所有 epoch 结束后，输出最佳 AUC 时的所有指标
    print("\n===== Final Best AUC Metrics =====")
    print(f"Best AUC = {best_auc:.4f} at Epoch {best_auc_epoch}")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    # train loss
    my_draw_loss(train_losses, loss_fig_name, result_folder)


if __name__ == '__main__':

    pid = os.getpid()
    print(f"当前代码运行的进程号是: {pid}")

    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0, help='0:MVFDSP')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=750, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=5*1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=13000, help='Number of epoch')
    parser.add_argument('--knn', type=int, required=False, default=10, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='features dimensions of drugs and side effects')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')
    parser.add_argument('--tenfold', action='store_true', default=True, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')

    args = parser.parse_args()

    modeling = [MVFDSP][args.model]
    metric = ['cosine', 'jaccard', 'euclidean'][args.metric]
    train_batch = args.train_batch
    lr = args.lr
    knn = args.knn
    num_epoch = args.epoch
    weight_decay = args.wd
    lamb = args.lamb
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    tenfold = args.tenfold
    save_model = args.save_model
    DF = args.DF
    not_FC = args.not_FC
    output_dim = args.dim
    eps = args.eps
    pca = args.PCA

    # 加载预处理数据
    processed_mask_mat = mask_mat_file
    if not os.path.isfile(processed_mask_mat):
        print('Missing data_WS files, generating......')
        generateMat()

    ######################################################################################
    result_folder = './result/'

    if tenfold:
        result_folder += '10WS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
    else:
        result_folder += '1WS_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)

    isExist = os.path.exists(result_folder)
    if not isExist:
        os.makedirs(result_folder)
    else:
        shutil.rmtree(result_folder)
        os.makedirs(result_folder)

    # 保存模型的文件
    modelsFolder = result_folder + '/model_save/'
    isCheckpointExist = os.path.exists(modelsFolder)
    if not isCheckpointExist:
        os.makedirs(modelsFolder)

    # 保存auc的结果
    for i in range(10):
        auc_result_folder = result_folder + f'/auc_result/fold{i+1}'
        isCheckpointExist = os.path.exists(auc_result_folder)
        if not isCheckpointExist:
            os.makedirs(auc_result_folder)

    # 保存scc的结果
    for i in range(10):
        scc_result_folder = result_folder + f'/scc_result/fold{i + 1}'
        isCheckpointExist = os.path.exists(scc_result_folder)
        if not isCheckpointExist:
            os.makedirs(scc_result_folder)

    # 保存每一个epoch的结果
    epoch_metrics_folder = result_folder + '/epoch_metrics'
    isCheckpointExist = os.path.exists(epoch_metrics_folder)
    if not isCheckpointExist:
        os.makedirs(epoch_metrics_folder)

    # 创建保存预测结果和标签的文件夹
    pred_result_folder = result_folder + '/pred_result'
    isCheckpointExist = os.path.exists(pred_result_folder)
    if not isCheckpointExist:
        os.makedirs(pred_result_folder)

    ######################################################################################

    result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']

    with open(result_log, 'w', newline='') as f:
        fieldnames = ['fold', 'pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'MAP', 'nDCG',
                      'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()



    start = datetime.datetime.now()
    # (id, frequencyMat) = next(split_data())
    for (id, frequencyMat, mask) in split_data(tenfold):

        start_ = datetime.datetime.now()
        main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
             frequencyMat, id + 1, mask, result_folder, save_model, DF, not_FC, output_dim, eps, pca)

        end_ = datetime.datetime.now()
        print('本次运行时间：{}\t'.format(end_ - start_))
    end = datetime.datetime.now()

    # 写入均值
    data = pd.read_csv(result_log)
    L = len(data.rMSE)
    avg = [sum(data.pearson) / L, sum(data.rMSE) / L, sum(data.spearman) / L, sum(data.MAE) / L,
           sum(data.auc_all) / L, sum(data.aupr_all) / L, sum(data.MAP) / L, sum(data.nDCG) / L,
           sum(data.P1) / L, sum(data.P5) / L, sum(data.P10) / L, sum(data.P15) / L,
           sum(data.R1) / L, sum(data.R5) / L, sum(data.R10) / L, sum(data.R15) / L]
    avg = [round(x, 5) for x in avg]
    print('\n\tavg pearson: {:.5f}\tavg rMSE: {:.5f}\tavg spearman: {:.5f}\tavg MAE: {:.5f}'.format(avg[0], avg[1],
                                                                                                    avg[2], avg[3]))
    print('\tavg all AUC: {:.5f}\tavg all AUPR: {:.5f}\tavg MAP: {:.5f}\tavg nDCG@10: {:.5f}'.format(avg[4], avg[5],
                                                                                                     avg[6], avg[7]))

    print('\tavg P@1: {:.5f}\tavg P@5: {:.5f}\tavg P@10: {:.5f}\tavg P@15: {:.5f}'.format(avg[8], avg[9], avg[10],
                                                                                          avg[11]))
    print('\tavg R@1: {:.5f}\tavg R@5: {:.5f}\tavg R@10: {:.5f}\tavg R@15: {:.5f}'.format(avg[12], avg[13], avg[14],
                                                                                          avg[15]))

    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        avg.insert(0, 'avg')
        writer.writerow(avg)

    print('运行时间：{}\t'.format(end - start))
