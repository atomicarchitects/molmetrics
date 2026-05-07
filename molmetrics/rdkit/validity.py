import os
import tempfile
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


def check_molecule_validity_with_smiles(mol: Chem.Mol, removeHs: bool = False) -> bool:
    """Checks whether a molecule is valid using the SMILES-based protocol from ADiT/Zatom-1.

    Writes molecule to PDB via pymatgen (which infers bonds from geometry),
    reads it back with RDKit, and checks if a canonical SMILES can be generated.
    This matches the evaluation protocol used by Joshi et al. (2025) and Luo et al. (2025).

    Args:
        mol: RDKit molecule with a 3D conformer and atom types (no bonds required).
        removeHs: Whether to remove hydrogens when reading the PDB file.

    Returns:
        True if a valid SMILES string can be generated.
    """
    try:
        from pymatgen.core import Molecule as PymatgenMolecule
    except ImportError:
        raise ImportError(
            "pymatgen is required for SMILES-based validity. "
            "Install it with: pip install pymatgen"
        )

    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    # Extract atom types and positions.
    species = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    coords = [list(conf.GetAtomPosition(i)) for i in range(n)]

    # Convert to pymatgen Molecule and write to PDB.
    pmg_mol = PymatgenMolecule(species=species, coords=coords)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        pdb_path = f.name
    try:
        pmg_mol.to(pdb_path, fmt="pdb")

        # Read back with RDKit (which parses the PDB bond information).
        rdkit_mol = Chem.MolFromPDBFile(pdb_path, removeHs=removeHs)
        if rdkit_mol is None:
            return False

        # Check if SMILES can be generated.
        smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=True)
        if smiles is None or smiles == "":
            return False

        return True
    except Exception:
        return False
    finally:
        os.unlink(pdb_path)