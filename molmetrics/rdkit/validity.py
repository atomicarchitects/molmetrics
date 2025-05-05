from typing import Sequence

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
from rdkit.Chem import rdBase

blocker = rdBase.BlockLogs()
RDLogger.DisableLog('rdApp.*')


def add_bonds(mol: Chem.Mol) -> Chem.Mol:
    """Adds bonds to a molecule."""
    mol = Chem.RWMol(mol)
    rdDetermineBonds.DetermineBonds(
        mol, charge=0, useHueckel=False, allowChargedFragments=True
    )
    return mol


def check_molecule_validity(mol: Chem.Mol) -> bool:
    """Checks whether a molecule is valid using xyz2mol.

    This function checks whether xyz2mol can determine all bonds in a molecule, with a net charge of 0.
    """
    # Make a copy of the molecule.
    mol = Chem.Mol(mol)

    # We should only have one conformer.
    assert mol.GetNumConformers() == 1

    try:
        mol = add_bonds(mol)
    except (ValueError, IndexError):
        return False

    if mol.GetNumBonds() == 0:
        return False

    return True
