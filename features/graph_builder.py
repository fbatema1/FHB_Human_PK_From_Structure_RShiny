"""
graph_builder.py
================
Converts SMILES strings to PyTorch Geometric Data objects for GNN training.

Atom features (9):
  - Atomic number (one-hot, top 44 elements + other)
  - Degree (0-10)
  - Formal charge
  - Number of Hs
  - Aromaticity (bool)
  - In ring (bool)
  - Hybridization (SP, SP2, SP3, SP3D, SP3D2, other)

Bond features (4):
  - Bond type (single, double, triple, aromatic)
  - Conjugated (bool)
  - In ring (bool)
  - Stereo (none, any, Z, E)

Usage:
    from features.graph_builder import MolGraphBuilder
    builder = MolGraphBuilder()
    data = builder.smiles_to_graph('CCO', y_cl=np.log10(5.0), y_vd=np.log10(0.6))
    dataset = builder.build_dataset(smiles_list, cl_values, vd_values)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from typing import List, Optional

# ── Atom feature settings ─────────────────────────────────────────────────────
ATOM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
    'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
    'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]  # 44 elements + 'other'

HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_STEREO_LIST = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
]


def one_hot(value, choices: list) -> List[int]:
    """One-hot encode value against choices list. Last slot = 'other'."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom) -> List[float]:
    """Build atom feature vector."""
    return (
        one_hot(atom.GetSymbol(), ATOM_LIST) +           # 45
        one_hot(atom.GetDegree(), list(range(11))) +     # 12
        [atom.GetFormalCharge()] +                       # 1
        [atom.GetTotalNumHs()] +                         # 1
        [int(atom.GetIsAromatic())] +                    # 1
        [int(atom.IsInRing())] +                         # 1
        one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)  # 6
    )  # Total: 67 features


def bond_features(bond) -> List[float]:
    """Build bond feature vector."""
    bt = bond.GetBondType()
    return [
        int(bt == rdchem.BondType.SINGLE),
        int(bt == rdchem.BondType.DOUBLE),
        int(bt == rdchem.BondType.TRIPLE),
        int(bt == rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ] + one_hot(bond.GetStereo(), BOND_STEREO_LIST)  # 4 + 5 = Total: 10 features


# Computed feature dimensions
N_ATOM_FEATURES = len(atom_features(
    Chem.MolFromSmiles('C').GetAtomWithIdx(0)
))
N_BOND_FEATURES = len(bond_features(
    Chem.MolFromSmiles('CC').GetBondWithIdx(0)
))


class MolGraphBuilder:
    """
    Converts SMILES to PyTorch Geometric Data objects.

    Each Data object contains:
      x          : atom feature matrix (n_atoms x N_ATOM_FEATURES)
      edge_index : bond connectivity (2 x 2*n_bonds) — both directions
      edge_attr  : bond feature matrix (2*n_bonds x N_BOND_FEATURES)
      y          : target tensor [log10(CL), log10(Vd)]
      smiles     : original SMILES string
      valid      : bool indicating successful conversion
    """

    def __init__(self):
        self.n_atom_features = N_ATOM_FEATURES
        self.n_bond_features = N_BOND_FEATURES

    def smiles_to_graph(
        self,
        smiles: str,
        y_cl:   Optional[float] = None,
        y_vd:   Optional[float] = None,
    ) -> Optional[Data]:
        """
        Convert a single SMILES string to a PyG Data object.

        Args:
            smiles: SMILES string
            y_cl:   log10(CL) target value (optional)
            y_vd:   log10(Vd) target value (optional)

        Returns:
            PyG Data object, or None if SMILES is invalid
        """
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        # Atom features
        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Bond features — add both directions for each bond
        if mol.GetNumBonds() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, N_BOND_FEATURES), dtype=torch.float)
        else:
            edge_indices = []
            edge_attrs   = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bf = bond_features(bond)
                # Both directions
                edge_indices += [[i, j], [j, i]]
                edge_attrs   += [bf, bf]

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attrs,  dtype=torch.float)

        # Target
        targets = []
        if y_cl is not None: targets.append(float(y_cl))
        if y_vd is not None: targets.append(float(y_vd))
        y = torch.tensor(targets, dtype=torch.float) if targets else None

        data = Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            smiles     = smiles,
        )
        if y is not None:
            data.y = y

        return data

    def build_dataset(
        self,
        smiles_list: List[str],
        cl_values:   Optional[List[float]] = None,
        vd_values:   Optional[List[float]] = None,
        verbose:     bool = True,
    ) -> List[Data]:
        """
        Build a list of PyG Data objects from a list of SMILES.

        CL and Vd values are log10-transformed before storing as targets.

        Returns:
            List of valid Data objects (invalid SMILES silently skipped)
        """
        dataset = []
        skipped = 0

        for i, smi in enumerate(smiles_list):
            y_cl = np.log10(cl_values[i]) if cl_values is not None and cl_values[i] > 0 else None
            y_vd = np.log10(vd_values[i]) if vd_values is not None and vd_values[i] > 0 else None

            data = self.smiles_to_graph(smi, y_cl=y_cl, y_vd=y_vd)
            if data is None:
                skipped += 1
                continue
            dataset.append(data)

        if verbose:
            print(f"[MolGraphBuilder] Built {len(dataset)} graphs "
                  f"({skipped} skipped — invalid SMILES)")

        return dataset

    @property
    def feature_dims(self):
        return {
            'n_atom_features': self.n_atom_features,
            'n_bond_features': self.n_bond_features,
        }
