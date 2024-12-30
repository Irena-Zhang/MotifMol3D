import os
import dgl
import json
from torch.utils.data import Dataset
import numpy as np
import torch
from motif_generator import *
import json


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


motif_g = motif_generator()
motif_g.get_motif_dict
with open('./para/cpd_TDB_des.json', 'r') as f:
    feature_3d_dict = json.loads(f.read())


class pred_data(Dataset):
    def __init__(self, graph=None, smiles=None):
        self.smiles = smiles
        self.graph = graph
        self.lens = len(smiles)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        property_s = self.smiles[item][1]
        property_s = property_s.strip().split(',')
        property = np.zeros((1, 11))
        for prop in property_s:
            property[0, int(prop)] = 1
        # property = np.zeros((1, 146))
        # for prop in property_s:
        #     property[0, int(prop)] = 1
        property = torch.tensor(property, dtype=torch.float32)
        motif_vacob = motif_g.motif_id_get(self.smiles[item][0], 7)
        maccs = Macss_get(self.smiles[item][0])
        maccs = torch.tensor(maccs, dtype=torch.float32)
        feature_3d = feature_3d_dict[self.smiles[item][0]]
        feature_3d = torch.tensor(feature_3d, dtype=torch.float32)
        # return self.smiles[item][0], self.graph[item], property, motif_vacob, maccs, feature_3d
        return self.smiles[item][0], self.graph, property, motif_vacob, maccs, feature_3d


class predict_data(Dataset):
    def __init__(self, graph=None, smiles=None, fea_3d_dict=None, motif_gene_smi=None):
        self.smiles = smiles
        self.graph = graph
        self.lens = len(smiles)
        self.fea_3d_dict = fea_3d_dict
        self.motif_gene_smi = motif_gene_smi

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        motif_vacob = self.motif_gene_smi.motif_id_get(self.smiles[item], 7)
        maccs = Macss_get(self.smiles[item])
        maccs = torch.tensor(maccs, dtype=torch.float32)
        feature_3d = self.fea_3d_dict[self.smiles[item]]
        feature_3d = torch.tensor(feature_3d, dtype=torch.float32)
        labels = np.zeros((1, 11))
        labels = torch.tensor(labels, dtype=torch.float32)

        # return self.smiles[item][0], self.graph[item], property, motif_vacob, maccs, feature_3d
        return self.smiles[item], self.graph, labels, motif_vacob, maccs, feature_3d
    

def predict_collate(samples):
    smiles, graphs, labels, motif_vocabs, maccs,feature_3d = map(list, zip(*samples))
    labels = torch.stack(labels)
    motif_vocabs = torch.stack(motif_vocabs)
    maccs = torch.stack(maccs)
    feature_3d = torch.stack(feature_3d)
    # bg = dgl.batch(graphs)
    bg = graphs
    return smiles, bg, labels, motif_vocabs, maccs, feature_3d


def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_trial_path(args):
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args


def get_configure(model):
    path = os.getcwd()
    p = os.path.join(path, "./para/gasa.json")
    with open(p, 'r') as f:
        config = json.load(f)
    return config


from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem import MACCSkeys


def Macss_get(smiles):

    max_MolMR, min_MolMR = -1000, 1000
    max_MolLogP, min_MolLogP = -1000, 1000
    max_MolWt, min_MolWt = -1000, 1000
    max_NumRotatableBonds, min_NumRotatableBonds = -1000, 1000
    max_NumAliphaticRings, min_NumAliphaticRings = -1000, 1000
    max_NumAromaticRings, min_NumAromaticRings = -1000, 1000
    max_NumSaturatedRings, min_NumSaturatedRings = -1000, 1000

    mol = Chem.MolFromSmiles(smiles)
    MACCS = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    MACCS_ids = np.zeros((20,))
    MACCS_ids[0] = Descriptors.MolMR(mol)
    MACCS_ids[1] = Descriptors.MolLogP(mol)
    MACCS_ids[2] = Descriptors.MolWt(mol)
    MACCS_ids[3] = Descriptors.NumRotatableBonds(mol)
    MACCS_ids[4] = Descriptors.NumAliphaticRings(mol)
    MACCS_ids[5] = MACCS[108]
    MACCS_ids[6] = Descriptors.NumAromaticRings(mol)
    MACCS_ids[7] = MACCS[98]
    MACCS_ids[8] = Descriptors.NumSaturatedRings(mol)
    MACCS_ids[9] = MACCS[137]
    MACCS_ids[10] = MACCS[136]
    MACCS_ids[11] = MACCS[145]
    MACCS_ids[12] = MACCS[116]
    MACCS_ids[13] = MACCS[141]
    MACCS_ids[14] = MACCS[89]
    MACCS_ids[15] = MACCS[50]
    MACCS_ids[16] = MACCS[160]
    MACCS_ids[17] = MACCS[121]
    MACCS_ids[18] = MACCS[149]
    MACCS_ids[19] = MACCS[161]

    for b in range(20):
        if b == 0:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 787.372
        elif b == 1:
            MACCS_ids[b] = (MACCS_ids[b] - -17.2699) / 804.642
        elif b == 2:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 787.372
        elif b == 3:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 787.372
        elif b == 4:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 787.372
        elif b == 6:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 787.372
        elif b == 8:
            MACCS_ids[b] = (MACCS_ids[b] - 0) / 14

    return MACCS_ids