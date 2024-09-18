import collections
from typing import List, Sequence, Optional

import numpy as np
from rdkit import Chem

from molmetrics.datatypes import LocalEnvironment, Atom, AtomWithPosition


def compute_local_environments(molecules: Sequence[Chem.Mol]) -> List[LocalEnvironment]:
    """Computes the local environments using bonded atoms."""

    def position_fn(mol: Chem.Mol):
        """Wrapper to get atom positions from atom indices."""

        def position(idx: int):
            return mol.GetConformer().GetAtomPosition(idx)

        return position

    def to_atom_fn(mol: Chem.Mol):
        """Wrapper to create Atoms from atom indices."""

        def to_atom(idx: int, pos: Optional[np.ndarray] = None):
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            if pos is None:
                return Atom(symbol)
            return AtomWithPosition(symbol, pos)

        return to_atom

    all_local_environments = []
    for mol in molecules:
        position = position_fn(mol)
        to_atom = to_atom_fn(mol)
        neighbors = collections.defaultdict(list)

        for bond in mol.GetBonds():
            atom1_index = bond.GetBeginAtomIdx()
            atom2_index = bond.GetEndAtomIdx()
            atom1_position = position(atom1_index)
            atom2_position = position(atom2_index)

            neighbors[atom1_index].append(
                to_atom(atom2_index, atom2_position - atom1_position)
            )
            neighbors[atom2_index].append(
                to_atom(atom1_index, atom1_position - atom2_position)
            )

        local_environments = [
            LocalEnvironment(
                to_atom(central_atom),
                tuple(sorted(neighbors[central_atom], key=lambda atom: atom.symbol)),
            )
            for central_atom in neighbors
        ]
        all_local_environments.extend(local_environments)

    return all_local_environments
