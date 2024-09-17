"""Metrics for evaluating generative models for molecules."""

from typing import Dict, Tuple, List, Optional, Sequence, Any
import logging

import os
import io
import glob
import collections
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np
import ase
import tqdm
from rdkit import RDLogger
import rdkit.Chem as Chem
from rdkit.Chem import rdDetermineBonds
import e3nn_jax as e3nn
import posebusters
import pandas as pd

log = logging.getLogger(__name__)



def get_all_valid_molecules(molecules: Sequence[Chem.Mol]) -> List[Chem.Mol]:
    """Returns all valid molecules (with bonds inferred)."""
    return [mol for mol in molecules if check_molecule_validity(mol)]


def get_all_valid_molecules_with_openbabel(
    molecules: Sequence[Tuple["openbabel.OBMol", "str"]],
) -> List["openbabel.OBMol"]:
    """Returns all molecules in a directory."""
    return [
        (mol, smiles)
        for mol, smiles in molecules
        if check_molecule_validity_with_openbabel(mol)
    ]


def get_all_molecules_with_openbabel(
    molecules_dir: str,
) -> List[Tuple["openbabel.OBMol", "str"]]:
    """Returns all molecules in a directory."""
    molecules = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        for mol in pybel.readfile("xyz", molecules_file):
            molecules.append((mol.OBMol, mol.write("smi").split()[0]))

    return molecules


def compute_molecule_sizes(molecules: Sequence[Chem.Mol]) -> np.ndarray:
    """Computes all of sizes of molecules."""
    return np.asarray([mol.GetNumAtoms() for mol in molecules])


def count_atom_types(
    molecules: Sequence[Chem.Mol], normalize: bool = False
) -> Dict[str, np.ndarray]:
    """Computes the number of atoms of each kind in each valid molecule."""
    atom_counts = collections.defaultdict(lambda: 0)
    for mol in molecules:
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

    if normalize:
        total = sum(atom_counts.values())
        atom_counts = {atom: count / total for atom, count in atom_counts.items()}

    return dict(atom_counts)


def compute_jensen_shannon_divergence(
    source_dist: Dict[str, float], target_dist: Dict[str, float]
) -> float:
    """Computes the Jensen-Shannon divergence between two distributions."""

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Computes the KL divergence between two distributions."""
        log_p = np.where(p > 0, np.log(p), 0)
        return (p * log_p - p * np.log(q)).sum()

    # Compute the union of the dictionary keys.
    # We assign a probability of 0 to any key that is not present in a distribution.
    keys = set(source_dist.keys()).union(set(target_dist.keys()))
    source_dist = np.asarray([source_dist.get(key, 0) for key in keys])
    target_dist = np.asarray([target_dist.get(key, 0) for key in keys])

    mean_dist = 0.5 * (source_dist + target_dist)
    return 0.5 * (
        kl_divergence(source_dist, mean_dist) + kl_divergence(target_dist, mean_dist)
    )


def compute_uniqueness(molecules: Sequence[Chem.Mol]) -> float:
    """Computes the fraction of valid molecules that are unique using SMILES."""
    all_smiles = []
    for mol in get_all_valid_molecules(molecules):
        smiles = Chem.MolToSmiles(mol)
        all_smiles.append(smiles)

    # If there are no valid molecules, return 0.
    if len(all_smiles) == 0:
        return 0.0

    return len(set(all_smiles)) / len(all_smiles)


def compute_uniqueness_with_openbabel(
    molecules: Sequence[Tuple["openbabel.OBMol", "str"]],
) -> float:
    """Computes the fraction of OpenBabel molecules that are unique using SMILES."""
    all_smiles = []
    for _, smiles in get_all_valid_molecules_with_openbabel(molecules):
        all_smiles.append(smiles)

    return len(set(all_smiles)) / len(all_smiles)


def compute_bond_lengths(
    molecules: Sequence[Chem.Mol],
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.
    Returns a dictionary where the key is the bond type, and the value is the list of all bond lengths of that bond.
    """
    bond_dists = collections.defaultdict(list)
    for mol in molecules:
        distance_matrix = Chem.Get3DDistanceMatrix(mol)

        if mol.GetNumBonds() == 0:
            raise ValueError("Molecule has no bonds.")

        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            atom_type_1, atom_type_2 = (
                min(atom_type_1, atom_type_2),
                max(atom_type_1, atom_type_2),
            )
            bond_length = distance_matrix[atom_index_1, atom_index_2]
            bond_dists[(atom_type_1, atom_type_2, bond_type)].append(bond_length)

    return {
        bond_type: np.asarray(bond_lengths)
        for bond_type, bond_lengths in bond_dists.items()
    }


def compute_local_environments(
    molecules: Sequence[Chem.Mol], max_num_molecules: int
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the number of distinct local environments given valid molecular geometries.
    Returns a dictionary where the key is the central atom, and the value is a dictionary of counts of distinct local environments.
    """
    local_environments = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )

    for mol_counter, mol in enumerate(molecules):
        if mol_counter == max_num_molecules:
            break

        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        for bond in mol.GetBonds():
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            counts[atom_index_1][atom_type_2] += 1
            counts[atom_index_2][atom_type_1] += 1

        for atom_index, neighbors in counts.items():
            central_atom_type = mol.GetAtomWithIdx(atom_index).GetSymbol()
            neighbors_as_string = ",".join(
                [f"{neighbor}{count}" for neighbor, count in sorted(neighbors.items())]
            )
            local_environments[central_atom_type][neighbors_as_string] += 1

        mol_counter += 1

    return {
        central_atom_type: dict(
            sorted(neighbors.items(), reverse=True, key=lambda x: x[1])
        )
        for central_atom_type, neighbors in local_environments.items()
    }
