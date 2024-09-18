from typing import Sequence, Callable, Dict, Tuple, List
import logging

import ase
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
import numpy as np

log = logging.getLogger(__name__)
try:
    import posebusters
except ImportError:
    log.warning("Posebusters not installed.")

from molmetrics.datatypes import Bond, Atom, LocalEnvironment
from molmetrics.molecules import Molecules
from molmetrics.rdkit import (
    io,
    validity,
    uniqueness,
    bond_lengths,
    bond_angles,
    local_environments,
)


class RDKitMolecules(Molecules):
    """Represents a collection of RDKit molecules."""

    def __init__(self, molecules: Sequence[Chem.Mol]):
        self._molecules = list(molecules)

    @property
    def molecules(self) -> List[Chem.Mol]:
        return self._molecules

    @classmethod
    def from_directory(
        self, directory: str, extension: str = ".xyz"
    ) -> "RDKitMolecules":
        """Loads molecules from a directory."""
        molecules = io.get_all_molecules(directory, extension)
        return RDKitMolecules(molecules)

    @classmethod
    def from_ase_atoms(self, atoms: Sequence[ase.Atoms]) -> "RDKitMolecules":
        """Loads molecules from ASE atoms."""
        molecules = io.ase_to_rdkit_molecules(atoms)
        return RDKitMolecules(molecules)

    def add_bonds(self) -> "RDKitMolecules":
        """Infers and adds bonds to the molecules."""
        return RDKitMolecules([validity.add_bonds(mol) for mol in self])

    def keep_if_true(self, function: Callable[[Chem.Mol], bool]) -> "RDKitMolecules":
        """Filters out molecules that do not satisfy a condition."""
        return RDKitMolecules([mol for mol in self if function(mol)])

    def keep_valid(self, verbose: bool = False) -> "RDKitMolecules":
        """Filters out invalid molecules."""
        if not verbose:
            RDLogger.DisableLog("rdApp.*")  # Disable RDKit logging.

        valid = RDKitMolecules(
            [mol for mol in self if validity.check_molecule_validity(mol)]
        )

        if not verbose:
            RDLogger.EnableLog("rdApp.*")  # Enable RDKit logging.
        return valid

    def keep_unique(self) -> "RDKitMolecules":
        """Filters out duplicate molecules."""
        return RDKitMolecules(uniqueness.get_all_unique_molecules(self))

    def keep_non_identical(self, other: "RDKitMolecules") -> "RDKitMolecules":
        """Filters out molecules that are identical to those in another collection."""
        unique_smiles = uniqueness.get_all_smiles(self)
        other_smiles = set(uniqueness.get_all_smiles(other))
        assert len(unique_smiles) == len(self)
        return RDKitMolecules(
            [
                mol
                for mol, smiles in zip(self, unique_smiles)
                if smiles not in other_smiles
            ]
        )

    def bond_lengths(self) -> Dict[Bond, np.ndarray]:
        """Computes the bond lengths."""
        return bond_lengths.compute_bond_lengths(self)

    def bond_angles(self) -> Dict[Dict[Atom, Tuple[Atom, Atom]], np.ndarray]:
        """Computes the bond angles around each atom."""
        return bond_angles.compute_bond_angles(self)

    def local_environments(self) -> List[LocalEnvironment]:
        """Computes the local environments."""
        return local_environments.compute_local_environments(self)

    def posebusters_analysis(self, full_report: bool = False):
        """Returns the analyses results from Posebusters (https://github.com/maabuu/posebusters)."""
        return posebusters.PoseBusters(config="mol").bust(
            mol_pred=self, full_report=full_report
        )
