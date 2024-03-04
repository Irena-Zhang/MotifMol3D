from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, BRICS
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
import numpy as np
from tqdm import tqdm
import json
import matplotlib.cm as    cm
import matplotlib
from IPython.display import SVG, display
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
# import cairosvg
from multiprocessing.pool import Pool
from scipy import stats
from collections import Counter
import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem
from pysmiles import read_smiles
from collections import defaultdict


ring_atomnum = {}


def remove_atomIdx(mol):
    ring_indice = list(Chem.GetSymmSSSR(mol))
    aro_ring_atom = []
    # no_aro_ring_atom = []
    comm_atom = []
    for ring in ring_indice:
        # print(list(ring))
        aromatic_atom_in_ring = 0
        for atom_ind in ring:
            if mol.GetAtomWithIdx(atom_ind).GetIsAromatic():
                aromatic_atom_in_ring += 1
        if aromatic_atom_in_ring == len(ring):
            aro_ring_atom.append(list(ring))


    aro_ring_bond = []
    bond_ids = []
    for ring_aro in aro_ring_atom:
        medi_bond = []
        bond_id = []
        for idx, atom_idx in enumerate(ring_aro):
            bond_id.extend([_.GetIdx() for _ in mol.GetAtomWithIdx(atom_idx).GetBonds()])
            if idx == len(ring_aro) - 1:
                medi_bond.append(mol.GetBondBetweenAtoms(atom_idx, ring_aro[0]).GetIdx())
            else:
                medi_bond.append(mol.GetBondBetweenAtoms(atom_idx, ring_aro[idx+1]).GetIdx())
        aro_ring_bond.append(medi_bond)
        bond_ids.append(list(set(bond_id)))

    remove_bond = list(set(sum(bond_ids, [])))
    new_bond = []
    for idx, bond_ring in enumerate(bond_ids):
        medi_bond = []
        for _ in bond_ring:
            if not _ in sum(aro_ring_bond, []):
                medi_bond.append(_)
        new_bond.append(medi_bond)

    return aro_ring_atom, comm_atom, bond_ids, remove_bond, new_bond


def mol_to_nx(mol):
    G = nx.Graph()
    aro_ring_atom, comm_atom, bond_ids, remove_bond, new_bond = remove_atomIdx(mol)

    # Construct existing points
    for atom in mol.GetAtoms():
        if atom.GetIdx() in sum(aro_ring_atom, []):
            continue
        else:
            G.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       formal_charge=atom.GetFormalCharge(),
                       chiral_tag=atom.GetChiralTag(),
                       hybridization=atom.GetHybridization(),
                       num_explicit_hs=atom.GetNumExplicitHs(),
                       is_aromatic=atom.GetIsAromatic())
    # Construct virtual points
    vir_node_id = []
    i = 1
    for atom_list in aro_ring_atom:
        sub_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=tuple(atom_list))
        if sub_smi in ring_atomnum:
            atomic_num = ring_atomnum[sub_smi]
        else:
            ring_atomnum[sub_smi] = len(list(ring_atomnum)) + 55
            atomic_num = ring_atomnum[sub_smi]

        G.add_node(mol.GetNumAtoms() + i,
                   atomic_num=atomic_num,
                   formal_charge=0,
                   chiral_tag=Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                   hybridization=Chem.rdchem.HybridizationType.SP2,
                   num_explicit_hs=0,
                   is_aromatic=True)
        vir_node_id.append(mol.GetNumAtoms() + i)
        i += 1

    for bond in mol.GetBonds():
        if bond.GetIdx() in remove_bond:
            continue
        else:
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType(),
                       bond_stereo=bond.GetStereo())

    # Construct connections between virtual atoms
    for i in range(0, len(bond_ids)-1):
        for j in range(i+1, len(bond_ids)):
            if set(bond_ids[i]).intersection(bond_ids[j]):
                G.add_edge(vir_node_id[i],
                           vir_node_id[j],
                           bond_type=Chem.rdchem.BondType.AROMATIC,
                           bond_stereo=Chem.rdchem.BondStereo.STEREONONE)

    # Construct connections between virtual atoms and adjacent atoms
    for i, nei_bond in enumerate(new_bond):
        for nei_id in nei_bond:
            if mol.GetBondWithIdx(nei_id).GetBeginAtomIdx() in aro_ring_atom[i]:
                flag = False
                for idx, aro_ring_atom_idx in enumerate(aro_ring_atom):
                    if mol.GetBondWithIdx(nei_id).GetEndAtomIdx() in aro_ring_atom_idx:
                        flag = idx+1
                        break
                if flag:
                    G.add_edge(vir_node_id[i],
                               vir_node_id[flag-1],
                               bond_type=mol.GetBondWithIdx(nei_id).GetBondType(),
                               bond_stereo=mol.GetBondWithIdx(nei_id).GetStereo())
                else:
                    G.add_edge(vir_node_id[i],
                               mol.GetBondWithIdx(nei_id).GetEndAtomIdx(),
                               bond_type=mol.GetBondWithIdx(nei_id).GetBondType(),
                               bond_stereo=mol.GetBondWithIdx(nei_id).GetStereo())
            else:
                flag = False
                for idx, aro_ring_atom_idx in enumerate(aro_ring_atom):
                    if mol.GetBondWithIdx(nei_id).GetBeginAtomIdx() in aro_ring_atom_idx:
                        flag = idx+1
                        break
                if flag:
                    G.add_edge(vir_node_id[i],
                               vir_node_id[flag-1],
                               bond_type=mol.GetBondWithIdx(nei_id).GetBondType(),
                               bond_stereo=mol.GetBondWithIdx(nei_id).GetStereo())
                else:
                    G.add_edge(vir_node_id[i],
                               mol.GetBondWithIdx(nei_id).GetBeginAtomIdx(),
                               bond_type=mol.GetBondWithIdx(nei_id).GetBondType(),
                               bond_stereo=mol.GetBondWithIdx(nei_id).GetStereo())
    return G


def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx
    bond_types = nx.get_edge_attributes(G, 'bond_type')
    bond_stereo = nx.get_edge_attributes(G, 'bond_stereo')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)
        mol.GetBondBetweenAtoms(ifirst, isecond).SetStereo(bond_stereo[first, second])


    Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
    return mol


def do_all(smiles, validate=False):
    mol = Chem.MolFromSmiles(smiles.strip())
    G = mol_to_nx(mol) 
    mol = nx_to_mol(G)
    return G, mol


def main():
    smi = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O'
    smi = 'NC(=O)N1c2ccccc2C2OC2c2ccccc21'
    with open(r'data/cpd_class_end.txt') as f:
        smile_list = [line.strip().split('\t')[0] for line in f]
    new_smi = []
    singsmi_newsmi = {}
    for i, singe_smi in enumerate(smile_list):
        print('{}/{}'.format(i, 5698))
        print(singe_smi)
        G, mol = do_all(singe_smi, validate=True)
        smiles = Chem.MolToSmiles(mol)
        print(smiles)
        new_smi.append(smiles)
        singsmi_newsmi[singe_smi] = smiles
    print(singsmi_newsmi)
    print(len(singsmi_newsmi))


if __name__ == '__main__':
    main()
