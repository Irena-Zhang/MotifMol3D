import networkx as nx
import math
import numpy as np
import copy
from tqdm import tqdm
from pysmiles import read_smiles
from rdkit import Chem
import torch
from collections import Counter
from rdkit.Chem import Draw


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


# with open(r'C:\Users\user\Desktop\hjg\学硕\特定化合物在体内代谢途径的预测\deepGCFX-main\data\cpd_class_end.txt') as f:
#     smile_list = [line.strip().split('\t')[0] for line in f]
#
with open(r'/data2/jghu/meta_pathway/MotifMol3D-main/data/cpd_class_end_R.txt') as f:
    smile_list = [line.strip().split('\t')[0] for line in f]
# with open(r'/home/jghu/meta_pathway/MotifMol3D-main/data/cpd_class_end.txt') as f:
#     smile_list = [line.strip().split('\t')[0] for line in f]


class motif_generator(object):
    def __init__(self):
        g_list=[]
        self.smiles = smile_list
        for i in self.smiles:
            graph = read_smiles(i)
            g_list.append(graph)
        self.g_list = g_list
        # self.mol_net = read_smiles(self.smiles)
        self.vocab = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.g_e_weight = {}

    @property
    def get_motif_dict(self):
        g_list = self.g_list
        for g in tqdm(range(len(g_list)), desc='Get motif dict', unit='graph'):
            clique_list = []
            # 将环和键分离并记录
            mcb = nx.cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]
            # print(g_list[g].edges)
            # print(mcb_tuple)
            edges = []
            edges_mcb = []
            for e in g_list[g].edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                    elif e[0] in set(c) or e[1] in set(c):
                        edges_mcb.append(e)
                if count == 0:
                    edges.append(e)
            # 记录分子中不属于环的边
            edges = list(set(edges))
            nodes_labels, g_smiles = self.get_node_labels(g)

            element = nx.get_edge_attributes(g_list[g], name="order")
            atoms = g_list[g].nodes

            for e in edges:
                weight = element[tuple(e)]
                edge = ((nodes_labels[e[0]], nodes_labels[e[1]]), weight)
                clique_id = self.add_to_vocab(edge)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g],element))
                ring = []
                for i in range(len(m)):
                    ring.append(nodes_labels[m[i]])
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1

            for e in clique_list:
                self.add_weight(e, g)

            c_list = tuple(set(clique_list))

            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e] += 1

            e_weight = {}
            for e in c_list:
                e_weight[e] = self.weight_vocab[(g, e)]/(len(edges) + len(mcb_tuple))
            self.g_e_weight[g_smiles] = e_weight

        for i in list(self.g_e_weight.keys()):
            for m in list(self.g_e_weight[i].keys()):
                self.g_e_weight[i][m] = self.g_e_weight[i][m] * (math.log((len(self.g_list) + 1) / self.node_count[m]))
        # self.motif_embedding('Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O')
        # print(self.whole_node_count)
        # print(list(self.whole_node_count.values()))
        tf_tdf = []
        for i in list(self.g_e_weight.keys()):
            for m in list(self.g_e_weight[i].keys()):
                tf_tdf.append(self.g_e_weight[i][m])
        tf_tdf_order = sorted(tf_tdf)
        max_id = len(tf_tdf_order) - 1

        # 调换tf-tdf值
        # for i in list(self.g_e_weight.keys()):
        #     for m in list(self.g_e_weight[i].keys()):
        #         id = max_id - tf_tdf_order.index(self.g_e_weight[i][m])
        #         self.g_e_weight[i][m] = (tf_tdf_order[id] - tf_tdf_order[0]) / (tf_tdf_order[max_id] - tf_tdf_order[0])

        for i in list(self.g_e_weight.keys()):
            for m in list(self.g_e_weight[i].keys()):
                # id = max_id - tf_tdf_order.index(self.g_e_weight[i][m])
                self.g_e_weight[i][m] = (self.g_e_weight[i][m] - tf_tdf_order[0]) / (tf_tdf_order[max_id] - tf_tdf_order[0])
        # print(len(self.vocab))
        # tf_tdf_1 = []
        # for i in list(self.g_e_weight.keys()):
        #     for m in list(self.g_e_weight[i].keys()):
        #         tf_tdf_1.append(self.g_e_weight[i][m])

        # 储存各分子中基序的数目
        # motif_mole = []
        # for i in list(self.g_e_weight.keys()):
        #     len_mole = len(self.g_e_weight[i])
        #     motif_mole.append(len_mole)

        # 统计各分子的基序种类数目
        # countDict = Counter(motif_mole)
        # a = []
        # b = []
        # for i in countDict:
        #     a.append(i)
        #     b.append(countDict[i])
        # print(a)
        # print(b)

    def motif_id_get(self,smiles,cut_off):
        motif_order_dict = sorted(self.g_e_weight[smiles].items(), key=lambda x: x[1], reverse=True)
        motif_order = list(id for id, tf_tdf in motif_order_dict)
        if len(motif_order) >= cut_off:
            motif_order = motif_order[0:cut_off]
            # print(motif_order)
        else:
            motif_order = motif_order + [400 for i in range(cut_off-len(motif_order))]
            # print(motif_order)

        return torch.tensor(motif_order, dtype=torch.long)

    def motif_embedding(self, smiles):
        motif_embedding = np.zeros((1, len(self.vocab)))
        for motif_id in list(self.g_e_weight[smiles].keys()):
            motif_embedding[0, int(motif_id)] = self.g_e_weight[smiles][motif_id]
        motif_embedding = torch.tensor(motif_embedding, dtype=torch.float32)

        return motif_embedding

    def get_node_labels(self,g):
        allowable_features = {
            'possible_atomic_num_list': list(range(0, 119))}
        atom_features_list = []
        mol = get_mol(self.smiles[g])
        for atom in mol.GetAtoms():
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())]
            atom_features_list.extend(atom_feature)

        return atom_features_list, self.smiles[g]

    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        for i in range(len(c)):
            if (c, weight) in self.vocab:
                return self.vocab[(c, weight)]
            else:
                c = self.shift_right(c)
                weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(list(self.vocab.keys()))

        return self.vocab[(c, weight)]

    def add_weight(self, node_id, g):
        if (g, node_id) not in self.weight_vocab:
            self.weight_vocab[(g, node_id)] = 1
        else:
            self.weight_vocab[(g, node_id)] += 1

    @staticmethod
    def shift_right(l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g, element):
        weight_list = []
        for i in range(len(ring) - 1):
            try:
                weight = element[tuple([ring[i], ring[i+1]])]
                weight_list.append(weight)
            except:
                weight = element[tuple([ring[i + 1], ring[i]])]
                weight_list.append(weight)

        try:
            weight = element[tuple([ring[-1], ring[0]])]
            weight_list.append(weight)
        except:
            weight = element[tuple([ring[0], ring[-1]])]
            weight_list.append(weight)

        return weight_list


if __name__ == '__main__':
    g = motif_generator().get_motif_dict
# smile_test = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O'
# smile_test1 = 'Nc1ncnc2c1ncn2'
# gh = motif_generator()
# gh.get_motif_dict
# gh.motif_id_get(smile_test, 6)
# embedding = gh.motif_embedding(smile_test)
# print(gh.vocab)
# print(embedding)
# print(len(gh.vocab))

