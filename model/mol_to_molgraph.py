from rdkit import Chem
from rdkit.Chem import Draw
from collections import defaultdict
import torch
import dgl
from dgl.data.utils import save_graphs
import pandas as pd
import torch.nn as nn
from functools import partial
from dgllife.utils import mol_to_bigraph
from dgllife.utils import ConcatFeaturizer, BaseAtomFeaturizer, BaseBondFeaturizer, atom_type_one_hot, atom_total_degree_one_hot, atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_implicit_valence_one_hot, atom_chiral_tag_one_hot, atom_is_aromatic, atom_is_in_ring
from dgllife.utils import bond_stereo_one_hot, atom_formal_charge_one_hot, atom_total_num_H_one_hot, bond_is_in_ring, bond_is_conjugated, bond_type_one_hot
from sklearn.preprocessing import LabelEncoder
import numpy as np
from smi_resconstract import do_all
# from smi_virtual_node_ring import do_all


fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
bond_vacob = defaultdict(lambda: len(bond_vacob))
atom_vacob = defaultdict(lambda: len(bond_vacob))
atom_dict = defaultdict(lambda: len(atom_dict))

allowable_features = {
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_stereo': [  # only for double bond stereo information
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
}


def create_atoms(mol):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # bond = bond_dict[[str(b.GetBondType()), str(b.GetIsAromatic(), str(b.GetIsConjugated()))]]
        bond = bond_dict[str(b.GetBondType()) + str(b.GetIsAromatic()) + str(b.GetIsConjugated())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))  # 相邻原子信息聚合 （原子类型，键价）
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)


def create_bond_fingerprints(he_str):
    b_list = []
    for b_str in he_str:
        bond_vacob[b_str]
        b_list.append(bond_vacob[b_str])

    return torch.LongTensor(b_list)


def create_atom_fingerprints(hv_str):
    b_list = []
    for h_str in hv_str:
        atom_vacob[h_str]
        b_list.append(bond_vacob[h_str])

    return torch.LongTensor(b_list)


def bond_type_get(bond):
    return [allowable_features['possible_bonds'].index(bond.GetBondType())]


def bond_sterero_get(bond):
    if bond.GetStereo() in allowable_features['possible_bond_stereo']:
        # print(bond.GetStereo())
        return [allowable_features['possible_bond_stereo'].index(bond.GetStereo())]
    else:
        # 其他立体信息均归为一类，则index归为4.
        return [4]


class AtomF(BaseAtomFeaturizer):
    """
    extract atom and bond feature
    """
    def __init__(self, atom_data_field='hv'):
        super(AtomF, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                # [partial(atom_type_one_hot, allowable_set=[
                #      'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                #  'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
                #  'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                #  'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'], encode_unknown=False),
                [atom_total_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge_one_hot,
                 atom_num_radical_electrons_one_hot,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_is_in_ring,
                 atom_total_num_H_one_hot,
                 atom_chiral_tag_one_hot])})


# class BondF(BaseBondFeaturizer):
#     def __init__(self, bond_data_field='he', self_loop=False):
#         super(BondF, self).__init__(
#             featurizer_funcs={bond_data_field: ConcatFeaturizer(
#                 [bond_type_one_hot,
#                  bond_is_conjugated,
#                  bond_is_in_ring,
#                  partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
#                                                              Chem.rdchem.BondStereo.STEREOANY,
#                                                              Chem.rdchem.BondStereo.STEREOZ,
#                                                              Chem.rdchem.BondStereo.STEREOE])])}, self_loop=self_loop)


class BondF(BaseBondFeaturizer):
    def __init__(self, bond_data_field='he', self_loop=False):
        super(BondF, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_get,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_sterero_get])}, self_loop=self_loop)


def update_node_bond_fea(g, vir_atom_id, vir_bond_id):
    node_fea = g.ndata['hv'].shape[1]
    bond_fea = g.edata['he'].shape[1]
    edges = list(g.edges())
    vir_index = []
    vir_index += [i for i, x in enumerate(edges[0].tolist()) if x in vir_atom_id]
    vir_index += [i for i, x in enumerate(edges[1].tolist()) if x in vir_atom_id]
    vir_index = list(set(vir_index))
    for atom_id in vir_atom_id:
        g.ndata['hv'][atom_id] = torch.zeros((1, node_fea), dtype=torch.float)
    for vir_idx in vir_index:
        g.edata['he'][vir_idx] = torch.zeros((1, bond_fea), dtype=torch.float)

    return None


def generate_graph(smiles, model, embed_bond, device):
    """
    Converts SMILES into graph with features.
    Parameters
    smiles: SMILES representation of the moelcule of interest
            type smiles: list
    return: DGL graph with features
            rtype: list
            
    """
    atom = AtomF(atom_data_field='hv')
    bond = BondF(bond_data_field='he', self_loop=True)
    graph = []
    graph_vir_atom = []
    for i in smiles:
        # i ='NC(=O)N1c2ccccc2C2OC2c2ccccc21'
        # mol = Chem.MolFromSmiles(i)
        # print(i)
        # G, mol, vir_atom_id, vir_bond_id = do_all(i)
        G, mol = do_all(i)
        # print(vir_atom_id)
        # print(vir_bond_id)
        # Draw.MolToFile(mol,'query.svg')
        # atoms = create_atoms(mol)
        # i_jbond_dict = create_ijbonddict(mol)
        # fingerprints = create_fingerprints(atoms, i_jbond_dict, 2)
        # fingerprints = torch.LongTensor(fingerprints).to(device)
        # x_atoms = model(fingerprints)
        # Chem.SanitizeMol(mol)
        g = mol_to_bigraph(mol,
                           node_featurizer=atom,
                           edge_featurizer=bond,
                           add_self_loop=True,
                           canonical_atom_order=False).to(device)
        
        # print(g.ndata['hv'].shape)
        # print(g.ndata['hv'].dtype)
        # exit()
        # g.vir_node = len(vir_atom_id)
        # print(g.ndata['hv'])
        # print(g.edata['he'])
        # g.ndata['hv'] = x_atoms
        # hv_str = list(map(lambda x: str(x), g.ndata['hv']))
        # atom_fingerprints = create_atom_fingerprints(hv_str).to(device)
        # x_atom = model(atom_fingerprints)
        # g.ndata['hv'] = x_atom
        # he_str = list(map(lambda x: str(x), g.edata['he']))
        # bond_fingerprints = create_bond_fingerprints(he_str).to(device)
        # x_bond = embed_bond(bond_fingerprints)
        # g.edata['he'] = x_bond
        # update_node_bond_fea(g, vir_atom_id, vir_bond_id)
        # print(g)
        # print(g.ndata['hv'].shape)
        # print(g.edata['he'].shape)
        # print(g.nodes())
        # print(g.edges())
        # exit()
        graph.append(g)
        # graph_vir_atom.append(vir_atom_id)
        # 应该没有问题，参数量增加有限，不影响计算速度
    # return graph, graph_vir_atom
    return graph


if __name__ == "__main__":
    # 测试是否是生成双向边
    atom = AtomF(atom_data_field='hv')
    bond = BondF(bond_data_field='he', self_loop=True)
    mol = Chem.MolFromSmiles('CC(=O)C(=O)O')

    g = mol_to_bigraph(mol,
                       node_featurizer=atom,
                       edge_featurizer=bond,
                       add_self_loop=True)
    print(g.ndata['hv'])
    print(g.edata['he'])
    print(g.edata['he'].shape)

