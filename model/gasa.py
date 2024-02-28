import json
import torch
import timeit
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score
import pandas as pd
from model.gasa_utils import generate_graph
from model.model import gasa_classifier
from model.data import pred_data, mkdir_p, init_trial_path, get_configure, predict_collate,shuffle_dataset,split_dataset
from model.hyper import init_hyper_space, EarlyStopping
from hyperopt import fmin, tpe
from copy import deepcopy
from argparse import ArgumentParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse():
    '''
    load Parameters
    '''
    parser = ArgumentParser(' Multi Classification')
    parser.add_argument('-n', '--num-epochs', type=int, default=80)
    parser.add_argument('-mo', '--model', default='GASA')
    parser.add_argument('-ne', '--num-evals', type=int, default=None,
                        help='the number of hyperparameter searches (default: None)')
    parser.add_argument('-me', '--metric', choices=['acc', 'loss', 'roc_auc_score'],
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-p', '--result-path', type=str, default='gasa/results',
                        help='Path to save training results (default: classification_results)')
    args = parser.parse_args().__dict__

    if args['num_evals'] is not None:
        assert args['num_evals'] > 0, 'Expect the number of hyperparameter search trials to ' \
                                        'be greater than 0, got {:d}'.format(args['num_evals'])
        print('Start hyperparameter search with Bayesian '
                'optimization for {:d} trials'.format(args['num_evals']))
        trial_path = bayesian_optimization(args)
    else:
        print('Use the manually specified hyperparameters')

    return args


def run_train_epoch(args, model, train_loader, device):
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
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_total = 0
    for iter, (smiles, bg, label) in enumerate(train_loader):
        label = torch.squeeze(label).to(device)
        bg, label = bg.to(device), label.to(device)
        prediction = model(bg)[0]
        loss = F.binary_cross_entropy(prediction, label)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()
        loss_total += loss.to('cpu').data.numpy()
        losses['loss'].append(loss)

    return loss_total, loss
    #     train_loss += loss.detach().item()
    #     pred = torch.max(prediction, 1)[1]
    #     pred_all.extend(pred.cpu().numpy())
    #     label_all.extend(label.cpu().numpy())
    #     acc += pred.eq(label.view_as(pred)).cpu().sum()
    # train_acc = acc.numpy() / len(train_loader.dataset)
    # train_loss /= (iter + 1)
    # return train_loss, train_acc


def run_val_epoch(args, model, val_loader, loss_func):
    model.eval()
    val_pred = []
    val_label = []
    pos_pro = []
    neg_pro = []
    with torch.no_grad():
        for iter, (smiles, bg) in enumerate(val_loader):
            pred = model(bg)[0]
            pos_pro += pred[:, 0].detach().cpu().numpy().tolist()
            neg_pro += pred[:, 1].detach().cpu().numpy().tolist()
            pred1 = torch.max(pred, 1)[1].view(-1)  
            val_pred += pred1.detach().cpu().numpy().tolist()
    return val_pred, pos_pro, neg_pro


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


def GASA(smiles):
    '''
    GASA model for prediction
    Parameters
    smiles: SMILES representation of the moelcule of interest
    return
    pred: predicted label for the given molecules (0:ES; 1:HS)
    pos: the postive probability
    neg: the negative probability 
    '''
    args = parse()
    torch.manual_seed(0)
    np.random.seed(0)
    exp_config = get_configure(args['model'])
    exp_config.update({'model': args['model']}) 
    args = init_trial_path(args)
    ls_smi = []
    if isinstance(smiles, list):  
        ls_smi = smiles
    else:
        ls_smi.append(smiles)
    graph = generate_graph(ls_smi)
    data = pred_data(graph=graph, smiles=ls_smi)             
    data_loader = DataLoader(data, batch_size=exp_config['batch_size'], shuffle=False, collate_fn=predict_collate)
    model = load_model(exp_config)
    loss_func = nn.CrossEntropyLoss()
    path = os.getcwd()
    pth = os.path.join(path, "model/gasa.pth")
    checkpoint = torch.load(pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    trial_path = args['result_path'] + '/1'
    pred, pos, neg = run_val_epoch(args, model, data_loader, loss_func)
    return pred, pos, neg
    

def bayesian_optimization():
    '''
    hyperparameter optmization
    '''
    args = parse()
    results = []
    candidate_hypers = init_hyper_space(args['model'])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams)

        if args['metric'] in ['roc_auc_score', 'val_acc']:
            val_metric_to_minimize = 1 - val_metric 
        else:
            val_metric_to_minimize = val_metric
        results.append((trial_path, val_metric_to_minimize))
        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args['num_evals'])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_val_metric


def random_data_get():
    # with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\GASA-master\data\cpd_class_end.txt') as f:
    #     data = [line.strip().split('\t') for line in f]
    with open(r'/home/jghu/meta_pathway/GASA-master/data/cpd_class_end.txt') as f:
        data = [line.strip().split('\t') for line in f]
    data = shuffle_dataset(data, 123)
    dataset_train, dataset_ = split_dataset(data, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    return dataset_train, dataset_dev, dataset_test


def data_pre(data):
    ls_smi = [smile_pro[0] for smile_pro in data]
    graph = generate_graph(ls_smi)
    data_loda = pred_data(graph=graph, smiles=data)
    data_loader = DataLoader(data_loda, batch_size=10, shuffle=False, collate_fn=predict_collate)

    return data_loader


def run_test_epoch(args, model, loader, device):
    model.eval()
    score_list, label_list, t_list = [], [], []
    for iter, (smiles, bg, label) in enumerate(loader):
        labels = torch.squeeze(label).to(device)
        bg, labels = bg.to(device), labels.to(device)
        prediction = model(bg)[0]
        zs = prediction.to('cpu').data.numpy()
        ts = labels.to('cpu').data.numpy()
        scores = list(map(lambda x: x, zs))
        labels = list(map(lambda x: (x >= 0.5).astype(int), zs))
        score_list = np.append(score_list, scores)
        label_list = np.append(label_list, labels)
        t_list = np.append(t_list, ts)

    auc = accuracy_score(t_list, label_list)
    precision = precision_score(t_list, label_list)
    recall = recall_score(t_list, label_list)

    return auc, precision, recall


if __name__ == '__main__':
    args = parse()
    torch.manual_seed(0)
    np.random.seed(0)
    exp_config = get_configure(args['model'])
    exp_config.update({'model': args['model']})
    args = init_trial_path(args)
    losses = {'loss': []}

    dataset_train, dataset_dev, dataset_test = random_data_get()
    train_loader = data_pre(dataset_train)
    dev_loader = data_pre(dataset_dev)
    test_loader = data_pre(dataset_test)

    model = load_model(exp_config).to(device)
    # path = os.getcwd()
    # pth = os.path.join(path, "model/gasa.pth")
    # checkpoint = torch.load(pth, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    trial_path = args['result_path'] + '/1'
    print('Training...')
    print('Epoch \t Time(sec) \t Loss_train \t AUC_dev \t AUC_test \t Precision \t Recall')
    best_recon_loss = 0.1
    for epoch in range(1,201):
        start = timeit.default_timer()
        loss_total, loss = run_train_epoch(args, model, train_loader, device)
        end = timeit.default_timer()
        auc_dev = run_test_epoch(args, model, dev_loader, device)[0]
        auc_test, precision, recall = run_test_epoch(args, model, test_loader, device)

        time = end - start
        print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (
            epoch, time, loss_total, auc_dev, auc_test, precision, recall))

        if auc_dev > best_recon_loss:
            best_recon_loss = auc_dev
            torch.save(model.state_dict(),'pretrained_models/gasa_epo200_loss.pkl')
            print('saving the current best model...')

    P_properties = []
    T_properties = []
    for iter, (smiles, bg, label) in enumerate(test_loader):
        labels = torch.squeeze(label).to(device)
        bg, labels = bg.to(device), labels.to(device)
        prediction = model(bg)[0]

        torch.set_printoptions(precision=2)

        p_properties = prediction.data.to('cpu').numpy()
        t_properties = labels.data.to('cpu').numpy()

        p_properties[p_properties < 0.5] = 0
        p_properties[p_properties >= 0.5] = 1

        P_properties.extend(p_properties)
        T_properties.extend(t_properties)
        # print(p_properties)
    P = np.zeros((1, 11))
    T = np.zeros((1, 11))
    for _ in P_properties:
        P = np.vstack((P, _))
    for _ in T_properties:
        T = np.vstack((T, _))

    for c in range(11):
        y_true = T[1:, c]
        y_pred = P[1:, c]

        auc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print('Class ' + str(c + 1) + ' statistics:')
        print('Accuracy %.4f, Precision %.4f, Recall %.4f\n' % (auc, precision, recall))