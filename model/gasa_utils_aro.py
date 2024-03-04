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
                fingerprint = (vertices[i], tuple(sorted(neighbors)))  
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
        return [allowable_features['possible_bond_stereo'].index(bond.GetStereo())]
    else:
        return [4]


class AtomF(BaseAtomFeaturizer):
    """
    extract atom and bond feature
    """
    def __init__(self, atom_data_field='hv'):
        super(AtomF, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
     
                [atom_total_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge_one_hot,
                 atom_num_radical_electrons_one_hot,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_is_in_ring,
                 atom_total_num_H_one_hot,
                 atom_chiral_tag_one_hot])})

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
        G, mol = do_all(i)
        g = mol_to_bigraph(mol,
                           node_featurizer=atom,
                           edge_featurizer=bond,
                           add_self_loop=True,
                           canonical_atom_order=False).to(device)
        
        graph.append(g)

    return graph


if __name__ == "__main__":
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

