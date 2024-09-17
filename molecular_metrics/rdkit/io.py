import io
import os
from typing import List, Sequence

import ase
from rdkit import Chem


def to_rdkit_molecule(molecules_file: str, extension: str) -> Chem.Mol:
    """Converts a molecule from xyz format to an RDKit molecule."""
    if extension == ".xyz":
        mol = Chem.MolFromXYZFile(molecules_file)
    elif extension == ".mol":
        mol = Chem.MolFromMolFile(molecules_file)
    elif extension == ".pdb":
        mol = Chem.MolFromPDBFile(molecules_file)
    elif extension == ".sdf":
        mol = Chem.SDMolSupplier(molecules_file)[0]
    else:
        raise ValueError(f"Unsupported extension: {extension}")
    return Chem.Mol(mol)


def get_all_molecules(molecules_dir: str, extension: str) -> List[Chem.Mol]:
    """Returns all molecules in a directory with a given extension."""
    molecules = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(extension):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        mol = to_rdkit_molecule(molecules_file, extension)
        molecules.append(mol)
    return molecules


def ase_to_rdkit_molecule(ase_mol: ase.Atoms) -> Chem.Mol:
    """Converts a molecule from ase format to an RDKit molecule."""
    with io.StringIO() as f:
        ase.io.write(f, ase_mol, format="xyz")
        f.seek(0)
        xyz = f.read()
    mol = Chem.MolFromXYZBlock(xyz)
    return Chem.Mol(mol)


def ase_to_rdkit_molecules(ase_mols: Sequence[ase.Atoms]) -> List[Chem.Mol]:
    """Converts molecules from ase format to RDKit molecules."""
    return [ase_to_rdkit_molecule(mol) for mol in ase_mols]
