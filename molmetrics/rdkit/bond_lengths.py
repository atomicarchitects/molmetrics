from typing import Dict, Sequence
import collections
import logging

import jax
from rdkit import Chem
import numpy as np

from molmetrics.datatypes import Atom, Bond

log = logging.getLogger(__name__)


def compute_bond_lengths(molecules: Sequence[Chem.Mol]) -> Dict[Bond, np.ndarray]:
    """Computes the bond lengths."""
    bond_dists = collections.defaultdict(list)
    for mol in molecules:
        if mol.GetNumBonds() == 0:
            log.warning(f"Molecule {Chem.MolToSmiles(mol)} has no bonds.")
            continue

        distance_matrix = Chem.Get3DDistanceMatrix(mol)
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            atom_type_1, atom_type_2 = (
                min(atom_type_1, atom_type_2),
                max(atom_type_1, atom_type_2),
            )
            bond_length = distance_matrix[atom_index_1, atom_index_2]
            bond = Bond(
                Atom(atom_type_1),
                Atom(atom_type_2),
                bond_type,
            )
            bond_dists[bond].append(bond_length)

    return jax.tree_map(np.asarray, bond_dists, is_leaf=lambda x: isinstance(x, list))