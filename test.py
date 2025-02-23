import json
import torch
import timeit
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score,f1_score,matthews_corrcoef,hamming_loss,label_ranking_loss
import pandas as pd
from model.gasa_utils_aro import generate_graph
from model.aro_model_metric  import gasa_classifier
from model.data import pred_data,predict_data, mkdir_p, init_trial_path, get_configure, predict_collate,shuffle_dataset,split_dataset
from model.data import *
from copy import deepcopy
from argparse import ArgumentParser
import dgl
import rdkit
from rdkit import Chem
from model.motif_generator import motif_generator_smidd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import timedelta

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

pathway_mapping = {
    0: "Carbohydrate metabolism",
    1: "Energy metabolism",
    2: "Lipid metabolism",
    3: "Nucleotide metabolism",
    4: "Amino acid metabolism",
    5: "Metabolism of other amino acids",
    6: "Glycan biosynthesis and metabolism",
    7: "Metabolism of cofactors and vitamins",
    8: "Metabolism of terpenoids and polyketides",
    9: "Biosynthesis of other secondary metabolites",
    10: "Xenobiotics biodegradation and metabolism"
}

def Coverage(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
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
    parser.add_argument('--pretrained_weights', type=str, default="/data2/jghu/model/MotifMol3D-main/motif_3D_best.pkl")
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
    with open(r'./data/train_smi.txt') as f:
        data_left = [line.strip().split('\t') for line in f]
    with open(r'./data/test_smi.txt') as f:
        data_right = [line.strip().split('\t') for line in f]

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
    mcc = matthews_corrcoef(t_list, label_list)  # Calculate MCC
    ham_l = hamming_loss(t_list, label_list)
    coverage = Coverage(total_labels, total_preds)
    one_error = One_error(total_labels, total_preds)
    RL = Ranking_loss(total_labels, total_preds)

    return auc, precision, recall,f1_scroe, mcc, ham_l, coverage, one_error, RL
    # return auc, precision, recall,f1_scroe, ham_l, coverage, one_error, RL


class Pack_net(nn.Module):
    def __init__(self, n_fingerdict_node=5600, feature_dim_node=42,
                 n_fingerdict_bond=150, feature_dim_bond=6):
        super(Pack_net, self).__init__()
        self.embed = nn.Embedding(n_fingerdict_node, feature_dim_node)
        self.embed_edge = nn.Embedding(n_fingerdict_bond, feature_dim_bond)
        self.gasa_model = load_model(exp_config)

    def forward(self,):
        loss_total, loss = run_train_epoch(args, self.gasa_model, train_loader, device, self.embed, self.embed_edge)
        auc_dev = run_test_epoch(args, self.gasa_model, dev_loader, device, self.embed, self.embed_edge)[0]
        auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL = run_test_epoch(args, self.gasa_model, test_loader, device, self.embed,
                                                     self.embed_edge)

        return loss_total, auc_dev, auc_test, precision, recall,f1_scroe, ham_l, coverage, one_error, RL

def new_smi_prop(smi):
    with open(r'./data/cpd_class_end.txt') as f:
        smile_list_origin = [line.strip().split('\t')[0] for line in f]
        smile_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smile_list_origin]
    smile_list_orgin = smile_list_origin + [smi]
    smiles_list = smile_list + [smi]
    # motif_generator_smi = motif_generator_smidd(smile_list_orgin, smiles_list)
    motif_generator_smi = motif_generator_smidd(smiles_list)
    motif_generator_smi.get_motif_dict
    data_loader = predict_data(smiles=[smi], fea_3d_dict=feature_3d_dict, motif_gene_smi=motif_generator_smi)
    data_loader = DataLoader(data_loader, batch_size=1, shuffle=False, collate_fn=predict_collate)
    return data_loader

def predict_smi_end(data_loader):
    total_embed = torch.Tensor()
    for iter, (smiles, bg, label, motif_vocab, maccs, feature_3d) in enumerate(data_loader):
        bg = generate_graph(smiles, pack_model.embed, pack_model.embed_edge, device)
        bg = dgl.batch(bg)
        bg, motif_vocab, maccs, feature_3d = bg.to(device),  motif_vocab.to(device), maccs.to(device), feature_3d.to(device)
        embedding = pack_model.gasa_model(bg, motif_vocab, maccs, feature_3d)[2]
        total_embed = torch.cat((total_embed,embedding.cpu()),0)
        
    return total_embed

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

def xgb_result(embed_train, label_train,embed_test):
    embed_test = embed_test.detach().numpy() 
    xgboost_clf = XGBClassifier(n_estimators=300, max_depth=30)
    multi_target_xgb = MultiOutputClassifier(xgboost_clf, n_jobs=-1)
    multi_target_xgb.fit(embed_train, label_train)
    Y_test = multi_target_xgb.predict(embed_test)
    Y_test = torch.tensor(Y_test)
    
    return Y_test

def set_seed(seed=0):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    start_time = time.time()
    # Parse arguments
    args = parse()

    # Set random seed for reproducibility
    set_seed(0)

    # Load experiment configuration
    exp_config = get_configure(args['model'])
    exp_config.update({'model': args['model']})

    # Initialize trial path
    args = init_trial_path(args)
    
    # Initialize losses dictionary
    losses = {'loss': []}

    # Initialize model
    pack_model = Pack_net().to(device)

    # Load existing weights if specified
    if 'pretrained_weights' in args and os.path.exists(args['pretrained_weights']):
        # args['pretrained_weights'] = r'/data2/jghu/model/MotifMol3D-main/motif_3D_best.pkl'
        print(f"Loading weights from {args['pretrained_weights']}...")
        pack_model.load_state_dict(torch.load(args['pretrained_weights'], map_location=device))
    else:
        print("No pretrained weights specified or file does not exist. Initializing model from scratch.")
    
    # Prepare data
    print('data prepare ......')
    dataset_train, dataset_dev, dataset_test = random_data_get()
    train_loader = data_pre(dataset_train)

    # Extract features and labels
    embed_train, label_train = get_fea_labels(train_loader)

    # example :smiles
    # smi = 'O=C(O)[C@@H](O)[C@H](O)[C@H](O)CO'
    smi = 'CCCC(=O)O'
    print(f'input: {smi}')
    print('starting predict ......')
    predicted_loader = new_smi_prop(smi)
    embed_test = predict_smi_end(predicted_loader)
    # Perform XGBoost training and evaluation
    Y_test = xgb_result(embed_train, label_train,embed_test)
    print('predicted ......')
    # Y_test = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    indices = torch.nonzero(Y_test == 1)
    pathway_ids = [str(row[1].item()) for row in indices]
    print(f"pathway_ids: {pathway_ids}")
    pathways = [f"{pid}: {pathway_mapping.get(int(pid))}" for pid in pathway_ids]
    print(Y_test)
    if pathways:
        print(f"{smi} maybe participates in {pathways} pathway")
    else:
        print(f"{smi} may not be involved in any pathways in the database.")
    print('---------- GOOD LUCK ----------')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {timedelta(seconds=total_time)}")
