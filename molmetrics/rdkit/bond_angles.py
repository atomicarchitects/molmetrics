from typing import Dict, Sequence, Tuple
import collections
import logging

import jax
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np

from molmetrics.datatypes import Atom, Bond

log = logging.getLogger(__name__)


def compute_bond_angles(molecules: Sequence[Chem.Mol]) -> Dict[Dict[Atom, Tuple[Atom, Atom]], np.ndarray]:
    """Computes the bond angles around each atom."""
    bond_angles = collections.defaultdict(dict)
    for mol in molecules:
        if mol.GetNumBonds() == 0:
            log.warning(f"Molecule {Chem.MolToSmiles(mol)} has no bonds.")
            continue

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            for bond1 in atom.GetBonds():
                for bond2 in atom.GetBonds():
                    if bond1.GetIdx() > bond2.GetIdx():
                        continue
                    atom1_idx = bond1.GetOtherAtomIdx(atom_idx)
                    atom2_idx = bond2.GetOtherAtomIdx(atom_idx)
                    atom1_idx, atom2_idx = (
                        min(atom1_idx, atom2_idx),
                        max(atom1_idx, atom2_idx),
                    )
                    angle = rdMolTransforms.GetAngleDeg(mol.GetConformer(), atom1_idx, atom_idx, atom2_idx)
                    
                    atom1 = Atom(mol.GetAtomWithIdx(atom1_idx).GetSymbol())
                    atom2 = Atom(mol.GetAtomWithIdx(atom2_idx).GetSymbol())
                    bond_angles[atom][atom1, atom2] = angle

    return jax.tree_map(np.asarray, bond_angles, is_leaf=lambda x: isinstance(x, list))