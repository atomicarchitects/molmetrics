"""Atom and molecule stability metrics following EDM (Hoogeboom et al., 2022).

Determines bonds from pairwise distances and atom types using lookup tables,
then checks if each atom has the correct valency.

- Atom stability: fraction of atoms with correct valency
- Molecule stability: fraction of molecules where ALL atoms are stable

Bond lengths and margins match the EDM codebase exactly:
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/bond_analyze.py
"""

from typing import Tuple
import numpy as np
from rdkit import Chem


# Bond lengths in picometers from EDM.
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
BONDS1 = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
           'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
           'Cl': 127, 'Br': 141, 'I': 161},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
           'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
           'I': 214},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
           'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
    'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
           'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
           'I': 194},
    'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
           'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
           'I': 187},
    'B': {'H': 119, 'Cl': 175},
    'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
            'F': 160, 'Cl': 202, 'Br': 215, 'I': 243},
    'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
            'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
            'Br': 214},
    'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
           'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
           'I': 234},
    'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
            'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
    'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
           'S': 210, 'F': 156, 'N': 177, 'Br': 222},
    'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
           'S': 234, 'F': 187, 'I': 266},
    'As': {'H': 152},
}

BONDS2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186},
}

BONDS3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113},
}

# Fixed margins in picometers, matching EDM exactly.
MARGIN1 = 10  # single bond margin (pm)
MARGIN2 = 5   # double bond margin (pm)
MARGIN3 = 3   # triple bond margin (pm)

# Allowed valencies (total bond order) per element, matching EDM.
ALLOWED_VALENCIES = {
    'H': [1], 'C': [4], 'N': [3], 'O': [2], 'F': [1],
    'B': [3], 'Al': [3], 'Si': [4],
    'P': [3, 5], 'S': [4], 'Cl': [1],
    'As': [3], 'Br': [1], 'I': [1],
    'Hg': [1, 2], 'Bi': [3, 5],
}


def get_bond_order(atom1: str, atom2: str, distance_angstrom: float,
                   check_exists: bool = False) -> int:
    """Determine bond order from element types and distance.

    Matches EDM's get_bond_order exactly, using fixed pm margins.

    Args:
        atom1: Element symbol of first atom.
        atom2: Element symbol of second atom.
        distance_angstrom: Distance between atoms in Angstroms.
        check_exists: If True, return 0 for unknown atom pairs instead of raising.

    Returns:
        Bond order: 0 (no bond), 1 (single), 2 (double), or 3 (triple).
    """
    distance = 100 * distance_angstrom  # Convert Angstroms to picometers.

    if check_exists:
        if atom1 not in BONDS1:
            return 0
        if atom2 not in BONDS1[atom1]:
            return 0

    # Check single bond threshold first.
    if atom1 not in BONDS1 or atom2 not in BONDS1.get(atom1, {}):
        return 0

    if distance < BONDS1[atom1][atom2] + MARGIN1:
        # Check double bond.
        if atom1 in BONDS2 and atom2 in BONDS2.get(atom1, {}):
            thr_bond2 = BONDS2[atom1][atom2] + MARGIN2
            if distance < thr_bond2:
                # Check triple bond.
                if atom1 in BONDS3 and atom2 in BONDS3.get(atom1, {}):
                    thr_bond3 = BONDS3[atom1][atom2] + MARGIN3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


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
            bo = get_bond_order(elements[i], elements[j], dist, check_exists=True)
            bond_counts[i] += bo
            bond_counts[j] += bo

    # Check valency.
    n_stable = 0
    for i in range(n):
        allowed = ALLOWED_VALENCIES.get(elements[i])
        if allowed is None:
            # Unknown element — counts as unstable.
            continue
        if bond_counts[i] in allowed:
            n_stable += 1

    atom_stability = n_stable / n if n > 0 else 0.0
    molecule_stability = (n_stable == n)
    return atom_stability, molecule_stability