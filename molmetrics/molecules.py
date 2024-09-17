from typing import Dict, Tuple, List, Callable, Any
import abc

import numpy as np

from molmetrics import bispectrum
from molmetrics.datatypes import Bond, LocalEnvironment


class Molecules(abc.ABC):
    """Represents a collection of molecules."""

    def __len__(self) -> int:
        """Returns the number of molecules."""
        return len(self.molecules)

    def __iter__(self):
        """Returns an iterator over the molecules."""
        return iter(self.molecules)

    def __getitem__(self, index: int) -> "Molecule":
        """Returns a molecule."""
        return self.molecules[index]

    @property
    @abc.abstractmethod
    def molecules(self) -> List["Molecule"]:
        """Returns the molecules."""
        pass

    @classmethod
    @abc.abstractmethod
    def from_directory(cls, directory: str) -> "Molecules":
        """Loads molecules from a directory."""
        pass

    @abc.abstractmethod
    def keep_if_true(self, function: Callable[["Molecule"], bool]) -> "Molecules":
        """Filters out molecules that do not satisfy a condition."""
        pass

    @abc.abstractmethod
    def keep_valid(self) -> "Molecules":
        """Filters out invalid molecules."""
        pass

    def validity(self) -> float:
        """Computes the fraction of valid molecules."""
        return len(self.keep_valid()) / len(self)

    @abc.abstractmethod
    def keep_unique(self) -> "Molecules":
        """Filters out duplicate molecules."""
        pass

    def uniqueness(self) -> float:
        """Computes the fraction of unique molecules among valid molecules."""
        return len(self.keep_valid().keep_unique()) / len(self.keep_valid())

    @abc.abstractmethod
    def keep_non_identical(self, other: "Molecules") -> "Molecules":
        """Filters out molecules that are identical to those in another collection."""
        pass

    def non_identical(self, other: "Molecules") -> float:
        """Computes the fraction of identical molecules."""
        return len(self.keep_non_identical(other)) / len(self)

    @abc.abstractmethod
    def bond_lengths(self) -> Dict[Bond, np.ndarray]:
        """
        Computes the lengths for each type of chemical bond in given valid molecular geometries.
        Returns a dictionary where the key is the bond type, and the value is an array of all bond lengths of that bond.
        """

    @abc.abstractmethod
    def bond_angles(self) -> Dict[Tuple[Bond, Bond], np.ndarray]:
        """
        Computes the angles for each type of chemical bond in given valid molecular geometries.
        Returns a dictionary where the key is the tuple of bond types, and the value is an array of all bond angles between those bonds.
        """

    @abc.abstractmethod
    def local_environments(self) -> List[LocalEnvironment]:
        """Computes the local environments for all valid molecular geometries."""

    def bispectra(self) -> Dict[LocalEnvironment, np.ndarray]:
        """
        Computes the bispectra for all valid molecular geometries.
        Returns an array of the bispectra.
        """
        return {
            local_environment: bispectrum.compute_bispectrum_for_local_environment(local_environment)
            for local_environment in self.local_environments()
        }
