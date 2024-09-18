from typing import Sequence

from rdkit import Chem


def get_all_smiles(molecules: Sequence[Chem.Mol]) -> Sequence[str]:
    """Returns all SMILES strings."""
    return [Chem.MolToSmiles(mol) for mol in molecules]


def get_all_unique_molecules(molecules: Sequence[Chem.Mol]) -> Sequence[Chem.Mol]:
    """Returns all unique molecules."""
    all_smiles = get_all_smiles(molecules)
    unique_smiles = list(set(all_smiles))
    return [
        mol for mol, smiles in zip(molecules, all_smiles) if smiles in unique_smiles
    ]
