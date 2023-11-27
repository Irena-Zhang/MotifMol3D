import json
import torch
import timeit
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score,f1_score,hamming_loss,label_ranking_loss
import pandas as pd
from model.gasa_utils_aro import generate_graph
# from model.model import gasa_classifier, AtomEmbedding
from model.aro_model_metric  import gasa_classifier
from model.data import pred_data,predict_data, mkdir_p, init_trial_path, get_configure, predict_collate,shuffle_dataset,split_dataset
from model.hyper import init_hyper_space, EarlyStopping
from hyperopt import fmin, tpe
from copy import deepcopy
from argparse import ArgumentParser
import dgl
import rdkit
from rdkit import Chem
from motif_generator import motif_generator_smidd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Coverage(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    # print(label)
    D = len(label[0])
    N = len(label)
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    cover = 0
    for i in range(N):
        # Sorted from largest to smallest
        index = np.argsort(-output[i]).tolist()
        tmp = 0
        for item in label_index[i]:
            tmp = max(tmp, index.index(item))
        cover += tmp
    coverage = cover * 1.0 / N
    return coverage


def One_error(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    N = len(label)
    for i in range(N):
        if max(label[i]) == 0:
            print("This data is not in either category")
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    OneError = 0
    for i in range(N):
        if np.argmax(output[i]) not in label_index[i]:
            OneError += 1
    OneError = OneError * 1.0 / N
    return OneError


def Ranking_loss(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    RL = label_ranking_loss(label, output)
    return RL
def parse():
    '''
    load Parameters
    '''
    parser = ArgumentParser(' Binary Classification')
    parser.add_argument('-n', '--num-epochs', type=int, default=80)
    parser.add_argument('-mo', '--model', default='GASA')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='the number of hyperparameter searches (default: None)')
    parser.add_argument('-me', '--metric', choices=['acc', 'loss', 'roc_auc_score'],
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-p', '--result-path', type=str, default='gasa/results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__

    return args


def run_train_epoch(args, model, train_loader, device, embed_model, embed_bond):
    '''
    if retrain model
    Parameters
    epoch: Number of iterations
    model: the model for train
    train_loader: load the data for training
    loss_func: Loss function
    optimizer: The optimizer (Adam)
    return: 
    train_loss: compute loss
    train_acc: compute accurary
    '''
    pack_model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(pack_model.parameters(), lr=0.0005)
    loss_total = 0
    # print(pack_model)
    for iter, (smiles, bg, label, motif_vocab, maccs, feature_3d) in enumerate(train_loader):
        bg = generate_graph(smiles, embed_model, embed_bond, device)
        bg = dgl.batch(bg)
        label = torch.squeeze(label).to(device)
        bg, label, motif_vocab, maccs, feature_3d = bg.to(device), label.to(device), motif_vocab.to(device), maccs.to(
            device), feature_3d.to(device)
        prediction = model(bg, motif_vocab, maccs, feature_3d, )[0]
        prediction = torch.sigmoid(prediction)
        loss = F.binary_cross_entropy(prediction, label)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
        loss_total += loss.to('cpu').data.numpy()
        losses['loss'].append(loss)

    return loss_total, loss


def Find_Optimal_Cutoff(TPR, FPR, threshold): 
    '''
    Compute Youden index, find the optimal threshold
    Parameters
    TPR: True Positive Rate
    FPR: False Positive Rate
    threshold: 
    return
    optimal_threshold: optimal_threshold
    point: optimum coordinates
    '''
    y = TPR - FPR
    an = np.argwhere(y == np.amax(y))
    Youden_index = an.flatten().tolist()
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def statistic(y_true, y_pred):
    '''
    compute statistic results
    Parameters
    y_true: true label of the given molecules
    y_pred: predicted label 
    return
    tp: True Positive
    fn: False Negative
    fp: False Positive
    tn: True Negative
    '''
    c_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = list(c_mat.flatten())
    return tp, fn, fp, tn


def load_model(exp_configure):
    '''
    load GASA model
    '''
    if exp_configure['model'] == 'GASA':
        model = gasa_classifier(dropout=exp_configure['dropout'], 
                                num_heads=exp_configure['num_heads'], 
                                hidden_dim1=exp_configure['hidden_dim1'], 
                                hidden_dim2=exp_configure['hidden_dim2'], 
                                hidden_dim3=exp_configure['hidden_dim3'])
    else:
        return ValueError("Expect model 'GASA', got{}".format((exp_configure['model'])))
    return model


def random_data_get():
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\GASA-master\data\cpd_class_end.txt') as f:
    #     data = [line.strip().split('\t') for line in f]
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\deepGCFX-main\data\cpd_class_end.txt') as f:
    #     data = [line.strip().split('\t') for line in f]
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\GASA-master\data\class_subclass\class146.txt') as f:
    #     data = [line.strip().split('\t') for line in f]
    # with open(r'/home/jghu/meta_pathway/deepGCFX-main/data/cpd_class_end.txt') as f:
        # data = [line.strip().split('\t') for line in f]
    # with open(r'/home/jghu/meta_pathway/metabolic_path/kegg_classes.txt') as f:
    #     data = [line.strip().split('\t') for line in f]
    with open(r'/home/jghu/meta_pathway/GASA-master/data/train_smi.txt') as f:
        data_left = [line.strip().split('\t') for line in f]
    with open(r'/home/jghu/meta_pathway/GASA-master/data/test_smi.txt') as f:
        data_right = [line.strip().split('\t') for line in f]
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\GASA-master\data\train_smi.txt') as f:
    #     data_left = [line.strip().split('\t') for line in f]
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\GASA-master\data\test_smi.txt') as f:
        # data_right = [line.strip().split('\t') for line in f]
    # data = shuffle_dataset(data, 123)
    # dataset_train, dataset_ = split_dataset(data, 0.8)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    data = data_left + data_right
    data_train_val = data[0:4905]
    data_test = data[4905:]
    dataset_train_val = shuffle_dataset(data_train_val, 1234)
    dataset_train, dataset_dev = split_dataset(dataset_train_val, 0.9)
    dataset_test = shuffle_dataset(data_test, 1234)

    return dataset_train, dataset_dev, dataset_test


def data_pre(data):
    data_loda = pred_data(smiles=data)
    data_loader = DataLoader(data_loda, batch_size=256, shuffle=False, collate_fn=predict_collate)

    return data_loader


def run_test_epoch(args, model, loader, device, embed_model, embed_bond):
    pack_model.eval()
    score_list, label_list, t_list = [], [], []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    for iter, (smiles, bg, label, motif_vocab,maccs,feature_3d) in enumerate(loader):
        bg = generate_graph(smiles, embed_model, embed_bond, device)
        bg = dgl.batch(bg)
        label = torch.squeeze(label).to(device)
        bg, label, motif_vocab, maccs, feature_3d = bg.to(device), label.to(device), motif_vocab.to(device),maccs.to(device), feature_3d.to(
            device)
        prediction = model(bg, motif_vocab, maccs,feature_3d)[0]
        zs = torch.sigmoid(prediction).to('cpu').data.numpy()
        ts = label.to('cpu').data.numpy()
        scores = list(map(lambda x: x, zs))
        labels = list(map(lambda x: (x >= 0.5).astype(int), zs))
        score_list = np.append(score_list, scores)
        label_list = np.append(label_list, labels)
        t_list = np.append(t_list, ts)

        if label.shape[0] == 11:
            label = label.unsqueeze(0)
        total_preds = torch.cat((total_preds, prediction.cpu()), 0)
        total_labels = torch.cat((total_labels, label.cpu()), 0)

    auc = accuracy_score(t_list, label_list)
    precision = precision_score(t_list, label_list)
    recall = recall_score(t_list, label_list)
    f1_scroe = (2 * precision * recall) / (recall + precision)
    ham_l = hamming_loss(t_list, label_list)
    coverage = Coverage(total_labels, total_preds)
    one_error = One_error(total_labels, total_preds)
    RL = Ranking_loss(total_labels, total_preds)

    return auc, precision, recall,f1_scroe, ham_l, coverage, one_error, RL


class Pack_net(nn.Module):
    def __init__(self, n_fingerdict_node=5600, feature_dim_node=42,
                 n_fingerdict_bond=150, feature_dim_bond=6):
        super(Pack_net, self).__init__()
        self.embed = nn.Embedding(n_fingerdict_node, feature_dim_node)
        self.embed_edge = nn.Embedding(n_fingerdict_bond, feature_dim_bond)
        self.gasa_model = load_model(exp_config)

    def forward(self,):
        # print(list(pack_model.gasa_model.parameters())[0])
        # print(list(self.gasa_model.parameters())[0])
        loss_total, loss = run_train_epoch(args, self.gasa_model, train_loader, device, self.embed, self.embed_edge)
        auc_dev = run_test_epoch(args, self.gasa_model, dev_loader, device, self.embed, self.embed_edge)[0]
        auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL = run_test_epoch(args, self.gasa_model, test_loader, device, self.embed,
                                                     self.embed_edge)

        return loss_total, auc_dev, auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL
        # return self.gasa_model, self.embed, self.embed_edge


def new_smi_prop(smi):
    with open(r'/home/jghu/meta_pathway/deepGCFX-main/data/cpd_class_end.txt') as f:
        smile_list_origin = [line.strip().split('\t')[0] for line in f]
        smile_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smile_list_origin]
    smile_list_orgin = smile_list_origin + [smi]
    smiles_list = smile_list + [smi]
    motif_generator_smi = motif_generator_smidd(smile_list_orgin, smiles_list)
    motif_generator_smi.get_motif_dict
    data_loader = predict_data(smiles=[smi], fea_3d_dict=feature_3d_dict, motif_gene_smi=motif_generator_smi)
    data_loader = DataLoader(data_loader, batch_size=1, shuffle=False, collate_fn=predict_collate)
    # print(data_loader)
    return data_loader


def predict_smi_end(data_loader):
    for iter, (smiles, bg, label, motif_vocab, maccs, feature_3d) in enumerate(data_loader):
        bg = generate_graph(smiles, pack_model.embed, pack_model.embed_edge, device)
        bg = dgl.batch(bg)
        bg, motif_vocab, maccs, feature_3d = bg.to(device),  motif_vocab.to(device), maccs.to(device), feature_3d.to(device)
        prediction = pack_model.gasa_model(bg, motif_vocab, maccs, feature_3d)[0]

    return prediction


def get_fea_labels(loader):
    pack_model.eval()
    total_embed = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for iter, (smiles, bg, label, motif_vocab,maccs,feature_3d) in enumerate(loader):
            bg = generate_graph(smiles, pack_model.embed, pack_model.embed_edge, device)
            bg = dgl.batch(bg)
            label = torch.squeeze(label).to(device)
            bg, label, motif_vocab, maccs, feature_3d = bg.to(device), label.to(device), motif_vocab.to(
                device), maccs.to(device), feature_3d.to(
                device)
            if label.shape[0] == 11:
                label = label.unsqueeze(0)
            embedding = pack_model.gasa_model(bg, motif_vocab, maccs, feature_3d)[2]
            total_embed = torch.cat((total_embed,embedding.cpu()),0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_embed,total_labels


def xgb_result():
    print('xgb********************xgb')
    xgboost_clf = XGBClassifier(n_estimators=300, max_depth=30)
    multi_target_xgb = MultiOutputClassifier(xgboost_clf, n_jobs=-1)
    multi_target_xgb.fit(embed_train, label_train)
    Y_dev = multi_target_xgb.predict(embed_dev)
    Y_test = multi_target_xgb.predict(embed_test)
    Y_dev = torch.tensor(Y_dev)
    Y_test = torch.tensor(Y_test)

    acc_score, prec_score, rec_score, f11_score = 0., 0., 0., 0.
    ham_l = 0.
    print('dev------------')
    for i in range(label_dev.shape[0]):
        acc_score += accuracy_score(label_dev[i], Y_dev[i])
        prec_score += precision_score(label_dev[i], Y_dev[i])
        rec_score += recall_score(label_dev[i], Y_dev[i])
        f11_score += f1_score(label_dev[i], Y_dev[i])
        ham_l += hamming_loss(label_dev[i], Y_dev[i])

    acc_score = acc_score / label_dev.shape[0]
    prec_score = prec_score / label_dev.shape[0]
    rec_score = rec_score / label_dev.shape[0]
    f11_score = f11_score / label_dev.shape[0]
    ham_l_score = ham_l / label_dev.shape[0]
    coverage = Coverage(label_dev, Y_dev)
    one_error = One_error(label_dev, Y_dev)
    RL = Ranking_loss(label_dev, Y_dev)
    print(
        'Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%,\t, f1_score : %.4f%%,\t hl : %.4f%%, \t, coverage : %.4f%%,\t, one_error : %.4f%%,\t, RL : %.4f%%' % (
            acc_score, prec_score, rec_score, f11_score, ham_l_score, coverage, one_error, RL))

    acc_score, prec_score, rec_score, f11_score = 0., 0., 0., 0.
    ham_l = 0.
    print('test--------------')
    for i in range(label_dev.shape[0]):
        acc_score += accuracy_score(label_test[i], Y_test[i])
        prec_score += precision_score(label_test[i], Y_test[i])
        rec_score += recall_score(label_test[i], Y_test[i])
        f11_score += f1_score(label_test[i], Y_test[i])
        ham_l += hamming_loss(label_test[i], Y_test[i])

    acc_score = acc_score / label_test.shape[0]
    prec_score = prec_score / label_test.shape[0]
    rec_score = rec_score / label_test.shape[0]
    f11_score = f11_score / label_test.shape[0]
    ham_l_score = ham_l / label_test.shape[0]
    coverage = Coverage(label_test, Y_test)
    one_error = One_error(label_test, Y_test)
    RL = Ranking_loss(label_test, Y_test)

    print(
        'Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%,\t, f1_score : %.4f%%,\t hl : %.4f%%, \t, coverage : %.4f%%,\t, one_error : %.4f%%,\t, RL : %.4f%%' % (
            acc_score, prec_score, rec_score, f11_score, ham_l_score, coverage, one_error, RL))


def rf_result():
    print('RF********************RF')
    clf = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=60, random_state=0)
    multi_target_forest = MultiOutputClassifier(clf, n_jobs=-1)
    multi_target_forest.fit(embed_train, label_train)
    Y_dev = multi_target_forest.predict(embed_dev)
    Y_test = multi_target_forest.predict(embed_test)
    Y_dev = torch.tensor(Y_dev)
    Y_test = torch.tensor(Y_test)

    acc_score, prec_score, rec_score, f11_score = 0., 0., 0., 0.
    ham_l = 0.
    print('dev------------')
    for i in range(label_dev.shape[0]):
        acc_score += accuracy_score(label_dev[i], Y_dev[i])
        prec_score += precision_score(label_dev[i], Y_dev[i])
        rec_score += recall_score(label_dev[i], Y_dev[i])
        f11_score += f1_score(label_dev[i], Y_dev[i])
        ham_l += hamming_loss(label_dev[i], Y_dev[i])

    acc_score = acc_score / label_dev.shape[0]
    prec_score = prec_score / label_dev.shape[0]
    rec_score = rec_score / label_dev.shape[0]
    f11_score = f11_score / label_dev.shape[0]
    ham_l_score = ham_l / label_dev.shape[0]
    coverage = Coverage(label_dev, Y_dev)
    one_error = One_error(label_dev, Y_dev)
    RL = Ranking_loss(label_dev, Y_dev)
    print(
        'Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%,\t, f1_score : %.4f%%,\t hl : %.4f%%, \t, coverage : %.4f%%,\t, one_error : %.4f%%,\t, RL : %.4f%%' % (
            acc_score, prec_score, rec_score, f11_score, ham_l_score, coverage, one_error, RL))

    acc_score, prec_score, rec_score, f11_score = 0., 0., 0., 0.
    ham_l = 0.
    print('test------------')
    for i in range(label_dev.shape[0]):
        acc_score += accuracy_score(label_test[i], Y_test[i])
        prec_score += precision_score(label_test[i], Y_test[i])
        rec_score += recall_score(label_test[i], Y_test[i])
        f11_score += f1_score(label_test[i], Y_test[i])
        ham_l += hamming_loss(label_test[i], Y_test[i])

    acc_score = acc_score / label_test.shape[0]
    prec_score = prec_score / label_test.shape[0]
    rec_score = rec_score / label_test.shape[0]
    f11_score = f11_score / label_test.shape[0]
    ham_l_score = ham_l / label_test.shape[0]
    coverage = Coverage(label_test, Y_test)
    one_error = One_error(label_test, Y_test)
    RL = Ranking_loss(label_test, Y_test)

    print('Accuracy : %.4f%%, \t Precision : %.4f%%, \t, Recall : %.4f%%,\t, f1_score : %.4f%%,\t hl : %.4f%%, \t, coverage : %.4f%%,\t, one_error : %.4f%%,\t, RL : %.4f%%' % (
            acc_score, prec_score, rec_score, f11_score, ham_l_score, coverage, one_error, RL))


if __name__ == '__main__':
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\deepGCFX-main\data\cpd_class_end.txt') as f:
    #     data_all = [line.strip().split('\t')[0] for line in f]
    args = parse()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    # random.seed(0)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(0)

    exp_config = get_configure(args['model'])
    exp_config.update({'model': args['model']})
    args = init_trial_path(args)
    losses = {'loss': []}

    # model = load_model(exp_config).to(device)
    pack_model = Pack_net().to(device)
    dataset_train, dataset_dev, dataset_test = random_data_get()
    train_loader = data_pre(dataset_train, )
    dev_loader = data_pre(dataset_dev, )
    test_loader = data_pre(dataset_test,)

    # path = os.getcwd()
    # pth = os.path.join(path, "model/gasa.pth")
    # checkpoint = torch.load(pth, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    trial_path = args['result_path'] + '/1'
    print('Training...')
    print('Epoch \t Time(sec) \t Loss_train \t AUC_dev \t AUC_test \t Precision \t Recall\t f1 \t ham_l \t coverage \t One_erro \t RL')
    best_recon_loss = 0.1
    for epoch in range(1, 200+1):
        # if epoch == 3:
        #     exit()
        start = timeit.default_timer()
        # train_loader = data_pre(dataset_train)
        # dev_loader = data_pre(dataset_dev)
        # test_loader = data_pre(dataset_test)
        # loss_total, loss = run_train_epoch(args, model, train_loader, device)
        # auc_dev = run_test_epoch(args, model, dev_loader, device)[0]
        # auc_test, precision, recall = run_test_epoch(args, model, test_loader, device)
        loss_total, auc_dev, auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL = pack_model()
        # model, embed_model, embed_bond = pack_model()

        end = timeit.default_timer()
        time = end - start
        print('%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            epoch, time, loss_total, auc_dev, auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL))

        if auc_test > best_recon_loss:
            best_recon_loss = auc_test
            # torch.save(pack_model.state_dict(),'pretrained_models/aro_hot_39_5_true_correct_1227_motif7.pkl')
            # torch.save(pack_model.state_dict(),'R_motif_logs/aro_hot_39_5_true_correct_1227_motif7_voff.pkl')
            torch.save(pack_model.state_dict(),'R_motif_logs/aro_hot_42_6_true_correct_0401_motif7_reget_no_reconsmi_dim3_64_v_metric_RF.pkl')
            print('saving the current best model...')
            for name, parameters in pack_model.named_parameters():
                if name == 'gasa_model.predict.weight':
                    print(name, ':', parameters[0])

            embed_train,label_train = get_fea_labels(train_loader)
            embed_dev,label_dev = get_fea_labels(dev_loader)
            embed_test, label_test = get_fea_labels(test_loader)
            xgb_result()
            rf_result()    

            # if epoch >= 90:
            #         with open('model/atc_drg_1448_TDB.json', 'r') as f:
            #             feature_3d_dict = json.loads(f.read())
            #         drg_smi_pre = {}
            #         for smi, smi_tdb in feature_3d_dict.items():
            #             data_loader = new_smi_prop(smi)
            #             # print(data_loader)
            #             pre_result = predict_smi_end(data_loader)
            #             # print(pre_result)
            #             drg_smi_pre[smi] = pre_result
            #         with open('drg_kegg_smi_path_cls.json', 'w') as f:
            #             json.dump(drg_smi_pre, f)
            P_properties = []
            T_properties = []

            for iter, (smiles, bg, label, motif_vocab, maccs, feature_3d) in enumerate(test_loader):
                label = torch.squeeze(label).to(device)
                bg = generate_graph(smiles, pack_model.embed, pack_model.embed_edge, device)
                bg = dgl.batch(bg)
                bg, label, motif_vocab, maccs, feature_3d = bg.to(device), label.to(device), motif_vocab.to(device), maccs.to(
                    device), feature_3d.to(device)
                prediction = pack_model.gasa_model(bg, motif_vocab, maccs, feature_3d)[0]

                torch.set_printoptions(precision=2)

                p_properties = p_properties = torch.sigmoid(prediction).data.to('cpu').numpy()
                t_properties = label.data.to('cpu').numpy()
                if t_properties.shape[0] == 15:
                    t_properties = [t_properties]

                p_properties[p_properties < 0.5] = 0
                p_properties[p_properties >= 0.5] = 1

                P_properties.extend(p_properties)
                T_properties.extend(t_properties)
            # P = np.zeros((1, 11))
            # T = np.zeros((1, 11))
            P = np.zeros((1, 11))
            T = np.zeros((1, 11))
            for _ in P_properties:
                P = np.vstack((P, _))
            for _ in T_properties:
                # print('*************')
                # print(_)
                T = np.vstack((T, _))

            for c in range(11):
                y_true = T[1:, c]
                y_pred = P[1:, c]

                auc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)

                print('Class ' + str(c + 1) + ' statistics:')
                print('Accuracy %.4f, Precision %.4f, Recall %.4f\n' % (auc, precision, recall))

    P_properties = []
    T_properties = []

    for iter, (smiles, bg, label, motif_vocab, maccs, feature_3d) in enumerate(test_loader):
        label = torch.squeeze(label).to(device)
        bg = generate_graph(smiles, pack_model.embed, pack_model.embed_edge, device)
        bg = dgl.batch(bg)
        bg, label, motif_vocab, maccs, feature_3d = bg.to(device), label.to(device), motif_vocab.to(device), maccs.to(
            device), feature_3d.to(device)
        prediction = pack_model.gasa_model(bg, motif_vocab, maccs, feature_3d)[0]

        torch.set_printoptions(precision=2)

        p_properties = torch.sigmoid(prediction).data.to('cpu').numpy()
        t_properties = label.data.to('cpu').numpy()
        if t_properties.shape[0] == 15:
            t_properties = [t_properties]

        p_properties[p_properties < 0.5] = 0
        p_properties[p_properties >= 0.5] = 1

        P_properties.extend(p_properties)
        T_properties.extend(t_properties)
    # P = np.zeros((1, 11))
    # T = np.zeros((1, 11))
    P = np.zeros((1, 11))
    T = np.zeros((1, 11))
    for _ in P_properties:
        P = np.vstack((P, _))
    for _ in T_properties:
        # print('*************')
        # print(_)
        T = np.vstack((T, _))

    for c in range(11):
        y_true = T[1:, c]
        y_pred = P[1:, c]

        auc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print('Class ' + str(c + 1) + ' statistics:')
        print('Accuracy %.4f, Precision %.4f, Recall %.4f\n' % (auc, precision, recall))