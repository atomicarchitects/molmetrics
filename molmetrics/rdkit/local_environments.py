import collections
from typing import List, Sequence

from rdkit import Chem
from molmetrics.datatypes import LocalEnvironment, Atom, Bond


def compute_local_environments(molecules: Sequence[Chem.Mol]) -> List[LocalEnvironment]:
    """Computes the local environments using bonded atoms."""
    get_position = lambda idx: mol.GetConformer().GetAtomPosition(idx)
    neighbors = collections.defaultdict(list)
    neighbor_positions = collections.defaultdict(list)

    for mol in molecules:
        neighbor_positions = collections.defaultdict(list)
        for bond in mol.GetBonds():
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom1 = Atom(mol.GetAtomWithIdx(atom_index_1).GetSymbol())
            atom2 = Atom(mol.GetAtomWithIdx(atom_index_2).GetSymbol())

            neighbors[atom1].append(atom2)
            neighbors[atom2].append(atom1)
            neighbor_positions[atom1].append(get_position(atom_index_2) - get_position(atom_index_1))
            neighbor_positions[atom2].append(get_position(atom_index_1) - get_position(atom_index_2))

    local_environments = [
        LocalEnvironment(central_atom, central_atom_neighbors, central_atom_neighbor_positions)
        for central_atom, central_atom_neighbors, central_atom_neighbor_positions in zip(
            neighbors.keys(), neighbors.values(), neighbor_positions.values()
        )
    ]
    return local_environments

