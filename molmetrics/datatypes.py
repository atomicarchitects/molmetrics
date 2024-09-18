"""Defines data types used in the molmetrics package."""

import collections
from typing import NamedTuple, Tuple, Union

from rdkit import Chem
import numpy as np


class Atom(NamedTuple):
    """Represents an atom in a molecule."""

    symbol: str

    def __repr__(self) -> str:
        return f"Atom {self.symbol}"

class Bond(NamedTuple):
    """Represents a chemical bond."""

    atom1: Atom
    atom2: Atom
    bond_type: Chem.BondType


class AtomWithPosition(NamedTuple):
    """Represents an atom in a molecule with a position."""

    symbol: str
    position: np.ndarray

    def __repr__(self) -> str:
        return f"Atom {self.symbol} at {self.position}"

class LocalEnvironment(NamedTuple):
    """Represents a local chemical environment."""

    central_atom: Atom
    neighbors: Tuple[Union[AtomWithPosition, Atom], ...]

    def __repr__(self) -> str:
        sorted_neighbors = sorted(self.neighbors, key=lambda atom: atom.symbol)
        counts = collections.Counter(sorted_neighbors)
        return f"LocalEnvironment {self.central_atom.symbol}({','.join([f'{atom.symbol}{count}' for atom, count in counts.items()])})"