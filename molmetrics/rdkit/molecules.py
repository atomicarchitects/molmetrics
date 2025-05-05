from typing import Sequence, Callable, Dict, Tuple, List
import logging

import ase
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import rdBase
import numpy as np

log = logging.getLogger(__name__)
try:
    import posebusters
except ImportError:
    posebusters = None
    log.warning("Posebusters not installed.")

from molmetrics import bispectrum
from molmetrics.datatypes import Bond, Atom, LocalEnvironment
from . import (
    io,
    validity,
    bond_lengths,
    bond_angles,
    local_environments,
)


class RDKitMolecules:
    """Represents a collection of RDKit molecules."""

    def __init__(self, molecules: Sequence[Chem.Mol]):
        self._molecules = [Chem.Mol(mol) for mol in molecules]

    def __len__(self) -> int:
        """Returns the number of molecules."""
        return len(self.molecules)

    def __iter__(self):
        """Returns an iterator over the molecules."""
        return iter(self.molecules)

    def __getitem__(self, index: int) -> "RDKitMolecules":
        """Returns a molecule."""
        return self.molecules[index]

    def validity(self) -> float:
        """Computes the fraction of valid molecules."""
        return len(self.keep_valid()) / len(self)

    def uniqueness(self) -> float:
        """Computes the fraction of unique molecules among valid molecules."""
        valid_mols = self.keep_valid()
        if not valid_mols:
            return 0.0

        smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        uniqueness = len(set(smiles)) / len(smiles)
        return uniqueness


    def non_identical(self, other: "RDKitMolecules") -> float:
        """Computes the fraction of identical molecules."""
        return len(self.keep_non_identical(other)) / len(self)

    def local_environment_bispectra(self, lmax: int = 4) -> bispectrum.BispectraSamples:
        """Computes the bispectra for all local environments."""
        return bispectrum.BispectraSamples(
            {
                local_environment: bispectrum.compute_bispectrum_for_local_environment(
                    local_environment, lmax
                )
                for local_environment in self.local_environments()
            }
        )

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
            # Suppress RDKit warnings.
            blocker = rdBase.BlockLogs()

        valid = RDKitMolecules(
            [mol for mol in self if validity.check_molecule_validity(mol)]
        )

        if not verbose:
            # Re-enable RDKit warnings.
            del blocker

        return valid

    def bond_lengths(self) -> Dict[Bond, np.ndarray]:
        """Computes the bond lengths."""
        return bond_lengths.compute_bond_lengths(self)

    def bond_angles(self) -> Dict[Dict[Atom, Tuple[Atom, Atom]], np.ndarray]:
        """Computes the bond angles around each atom."""
        return bond_angles.compute_bond_angles(self)

    def local_environments(self) -> List[LocalEnvironment]:
        """Computes the local environments."""
        return local_environments.compute_local_environments(self)

    def analyse_with_posebusters(self, full_report: bool = False):
        """Returns the analyses results from Posebusters (https://github.com/maabuu/posebusters)."""
        if posebusters is None:
            raise ImportError(
                "Posebusters is not installed. Please install it to use this feature."
            )
        return posebusters.PoseBusters(config="mol").bust(
            mol_pred=self, full_report=full_report
        )
