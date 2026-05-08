from typing import Sequence, Callable, Dict, Tuple, List
import logging

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import rdBase
import numpy as np

log = logging.getLogger(__name__)

from molmetrics.datatypes import Bond, Atom, LocalEnvironment
from . import (
    io,
    validity,
    uniqueness,
    bond_lengths,
    bond_angles,
    local_environments,
    stability,
)


class RDKitMolecules:
    """Represents a collection of RDKit molecules."""

    def __init__(self, molecules: Sequence[Chem.Mol]):
        self._molecules = [Chem.Mol(mol) for mol in molecules]

    def __len__(self) -> int:
        """Returns the number of molecules."""
        return len(self._molecules)

    def __iter__(self):
        """Returns an iterator over the molecules."""
        return iter(self._molecules)

    def __getitem__(self, index: int) -> Chem.Mol:
        """Returns a molecule."""
        return self._molecules[index]

    @property
    def molecules(self) -> List[Chem.Mol]:
        return self._molecules

    def validity(self) -> float:
        """Computes the fraction of valid molecules using xyz2mol (rdDetermineBonds).

        This is the stricter validity check — requires successful bond inference
        from 3D coordinates with net charge 0.
        """
        return len(self.keep_valid()) / len(self)

    def validity_with_smiles(self, removeHs: bool = False) -> float:
        """Computes the fraction of valid molecules using the SMILES-based protocol.

        Matches the evaluation used by ADiT (Joshi et al., 2025) and Zatom-1 (Morehead et al., 2025):
        writes to PDB via pymatgen, reads back with RDKit, checks if canonical SMILES
        can be generated. Generally more lenient than xyz2mol validity.
        """
        n_valid = sum(
            1
            for mol in self
            if validity.check_molecule_validity_with_smiles(mol, removeHs=removeHs)
        )
        return n_valid / len(self)

    def uniqueness(self) -> float:
        """Computes the fraction of unique molecules among valid molecules."""
        valid_mols = self.keep_valid()
        valid_mols = valid_mols.add_bonds()
        if not valid_mols:
            return 0.0
        unique_mols = uniqueness.get_all_unique_molecules(valid_mols)
        return len(unique_mols) / len(valid_mols)

    def uniqueness_with_smiles(self, removeHs: bool = False) -> float:
        """Computes the fraction of unique molecules among SMILES-valid molecules.

        Uses the SMILES-based validity protocol (ADiT/Zatom-1) to determine valid
        molecules, then computes uniqueness via canonical SMILES strings.
        """
        from . import validity as val_mod

        smiles_set = set()
        n_valid = 0
        for mol in self:
            smiles = val_mod.get_smiles_if_valid(mol, removeHs=removeHs)
            if smiles is not None:
                n_valid += 1
                smiles_set.add(smiles)
        if n_valid == 0:
            return 0.0
        return len(smiles_set) / n_valid

    def atom_stability(self) -> float:
        """Computes the fraction of atoms with correct valency (EDM metric).

        Uses bond distance lookup tables to infer bonds, then checks if each
        atom has the allowed number of bonds for its element type.
        """
        total_atoms = 0
        stable_atoms = 0
        for mol in self:
            if mol.GetNumConformers() == 0:
                continue
            atom_stab, _ = stability.compute_stability_for_molecule(mol)
            n = mol.GetNumAtoms()
            stable_atoms += int(round(atom_stab * n))
            total_atoms += n
        return stable_atoms / total_atoms if total_atoms > 0 else 0.0

    def molecule_stability(self) -> float:
        """Computes the fraction of molecules where all atoms are stable (EDM metric)."""
        total = 0
        stable = 0
        for mol in self:
            if mol.GetNumConformers() == 0:
                continue
            _, mol_stab = stability.compute_stability_for_molecule(mol)
            total += 1
            if mol_stab:
                stable += 1
        return stable / total if total > 0 else 0.0

    def compute_metric(self, metric_fn: Callable[[Chem.Mol], any]) -> List[any]:
        """Computes the given metric across valid molecules."""
        valid_mols = self.keep_valid()
        valid_mols = valid_mols.add_bonds()
        return [metric_fn(mol) for mol in valid_mols]

    def non_identical(self, other: "RDKitMolecules") -> float:
        """Computes the fraction of identical molecules."""
        return len(self.keep_non_identical(other)) / len(self)

    def bond_lengths(self) -> Dict[Bond, np.ndarray]:
        """Computes the bond lengths."""
        return bond_lengths.compute_bond_lengths(self)

    def bond_angles(self) -> Dict[Dict[Atom, Tuple[Atom, Atom]], np.ndarray]:
        """Computes the bond angles around each atom."""
        return bond_angles.compute_bond_angles(self)

    def local_environments(self) -> List[LocalEnvironment]:
        """Computes the local environments."""
        return local_environments.compute_local_environments(self)

    def local_environment_bispectra(self, lmax: int = 4):
        """Computes the bispectra for all local environments.

        Requires jax and e3nn-jax to be installed.
        """
        from molmetrics import bispectrum

        return bispectrum.BispectraSamples(
            {
                local_environment: bispectrum.compute_bispectrum_for_local_environment(
                    local_environment, lmax
                )
                for local_environment in self.local_environments()
            }
        )

    def add_bonds(self) -> "RDKitMolecules":
        """Infers and adds bonds to the molecules."""
        return RDKitMolecules([validity.add_bonds(mol) for mol in self])

    def keep_if_true(self, function: Callable[[Chem.Mol], bool]) -> "RDKitMolecules":
        """Filters out molecules that do not satisfy a condition."""
        return RDKitMolecules([mol for mol in self if function(mol)])

    def keep_valid(self, verbose: bool = False) -> "RDKitMolecules":
        """Filters out invalid molecules."""
        if not verbose:
            blocker = rdBase.BlockLogs()

        valid = RDKitMolecules(
            [mol for mol in self if validity.check_molecule_validity(mol)]
        )

        if not verbose:
            del blocker

        return valid

    def keep_valid_with_smiles(self, removeHs: bool = False) -> "RDKitMolecules":
        """Filters out invalid molecules using the SMILES-based protocol.

        Returns molecules with bonds inferred by pymatgen/OpenBabel (via PDB round-trip),
        not the original bondless molecules.
        """
        valid = []
        for mol in self:
            result = validity.get_mol_with_bonds_if_valid(mol, removeHs=removeHs)
            if result is not None:
                valid.append(result[1])
        return RDKitMolecules(valid)

    @classmethod
    def from_directory(
        cls, directory: str, extension: str = ".xyz"
    ) -> "RDKitMolecules":
        """Loads molecules from a directory."""
        molecules = io.get_all_molecules(directory, extension)
        return cls(molecules)

    @classmethod
    def from_ase_atoms(cls, atoms) -> "RDKitMolecules":
        """Loads molecules from ASE Atoms objects.

        Requires ase to be installed.
        """
        molecules = io.ase_to_rdkit_molecules(atoms)
        return cls(molecules)

    def analyse_with_posebusters(self, full_report: bool = False, config: str = "mol"):
        """Returns the analysis results from PoseBusters.

        Requires posebusters to be installed.
        """
        try:
            import posebusters
        except ImportError:
            raise ImportError(
                "posebusters is not installed. Install it with: pip install posebusters"
            )
        import warnings
        import logging as _logging

        pb_logger = _logging.getLogger("posebusters")
        prev_level = pb_logger.level
        pb_logger.setLevel(_logging.CRITICAL)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = posebusters.PoseBusters(config).bust(
                mol_pred=self, full_report=full_report
            )

        pb_logger.setLevel(prev_level)
        return result
