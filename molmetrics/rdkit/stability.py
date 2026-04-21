"""Atom and molecule stability metrics following EDM (Hoogeboom et al., 2022).

Determines bonds from pairwise distances and atom types using lookup tables,
then checks if each atom has the correct valency.

- Atom stability: fraction of atoms with correct valency
- Molecule stability: fraction of molecules where ALL atoms are stable
"""

from typing import Dict, Tuple
import numpy as np
from rdkit import Chem


# Typical bond lengths in Angstroms: (single, double, triple)
# Keys are sorted tuples of element symbols.
BOND_LENGTHS = {
    ('C', 'C'): (1.54, 1.34, 1.20),
    ('C', 'H'): (1.09, None, None),
    ('C', 'N'): (1.47, 1.29, 1.16),
    ('C', 'O'): (1.43, 1.23, 1.13),
    ('C', 'F'): (1.35, None, None),
    ('F', 'F'): (1.42, None, None),
    ('F', 'H'): (0.92, None, None),
    ('F', 'N'): (1.36, None, None),
    ('F', 'O'): (1.42, None, None),
    ('H', 'H'): (0.74, None, None),
    ('H', 'N'): (1.01, None, None),
    ('H', 'O'): (0.96, None, None),
    ('N', 'N'): (1.45, 1.25, 1.10),
    ('N', 'O'): (1.40, 1.21, None),
    ('O', 'O'): (1.48, 1.21, None),
}

# Margins for single, double, triple bonds (as fraction of bond length).
# From EDM: m1=10%, m2=5%, m3=3%.
MARGINS = (0.10, 0.05, 0.03)

# Allowed valencies (total bond order) per element.
ALLOWED_VALENCIES = {
    'H': [1],
    'C': [4],
    'N': [3],
    'O': [2],
    'F': [1],
}


def get_bond_order(elem1: str, elem2: str, distance: float) -> int:
    """Determine bond order from element types and distance.

    Returns 0 (no bond), 1 (single), 2 (double), or 3 (triple).
    """
    key = tuple(sorted([elem1, elem2]))
    if key not in BOND_LENGTHS:
        return 0

    lengths = BOND_LENGTHS[key]

    # Check triple bond first (tightest margin).
    if lengths[2] is not None:
        if distance < lengths[2] + lengths[2] * MARGINS[2]:
            return 3

    # Check double bond.
    if lengths[1] is not None:
        if distance < lengths[1] + lengths[1] * MARGINS[1]:
            return 2

    # Check single bond.
    if lengths[0] is not None:
        if distance < lengths[0] + lengths[0] * MARGINS[0]:
            return 1

    return 0


def compute_stability_for_molecule(mol: Chem.Mol) -> Tuple[float, bool]:
    """Compute atom stability and molecule stability for one molecule.

    Args:
        mol: RDKit molecule with a 3D conformer.

    Returns:
        atom_stability: fraction of atoms with correct valency
        molecule_stability: True if all atoms are stable
    """
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    # Get element symbols and positions.
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])

    # Compute bond orders from distances.
    bond_counts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            bo = get_bond_order(elements[i], elements[j], dist)
            bond_counts[i] += bo
            bond_counts[j] += bo

    # Check valency.
    n_stable = 0
    for i in range(n):
        allowed = ALLOWED_VALENCIES.get(elements[i], [])
        if bond_counts[i] in allowed:
            n_stable += 1

    atom_stability = n_stable / n if n > 0 else 0.0
    molecule_stability = (n_stable == n)
    return atom_stability, molecule_stability