from typing import Sequence

from rdkit import Chem


def get_all_smiles(molecules: Sequence[Chem.Mol]) -> Sequence[str]:
    """Returns all SMILES strings."""
    return [Chem.MolToSmiles(mol) for mol in molecules]


def get_all_unique_molecules(molecules: Sequence[Chem.Mol]) -> Sequence[Chem.Mol]:
    """Returns all unique molecules."""
    all_smiles = get_all_smiles(molecules)
    seen_smiles = set()
    unique_molecules = []
    for mol, smiles in zip(molecules, all_smiles):
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        unique_molecules.append(mol)
    return unique_molecules
