"""Tests for molmetrics stability and validity metrics.

Uses 3 QM9 molecules from the test data directory.
QM9 ground truth molecules should have high stability and validity.
"""
import os
import pytest
import molmetrics as mm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "qm9")


@pytest.fixture
def qm9_mols():
    """Load QM9 test molecules."""
    mols = mm.RDKitMolecules.from_directory(DATA_DIR)
    assert len(mols) == 3
    return mols


@pytest.fixture
def qm9_mols_with_bonds(qm9_mols):
    """Load QM9 test molecules with inferred bonds."""
    return qm9_mols.add_bonds()


class TestStability:
    """Tests for EDM-style atom/molecule stability."""

    def test_atom_stability_perfect_for_qm9(self, qm9_mols):
        """QM9 ground truth molecules should have perfect atom stability."""
        atom_stab = qm9_mols.atom_stability()
        assert atom_stab == 1.0, f"Expected 100% atom stability, got {atom_stab:.2%}"

    def test_molecule_stability_perfect_for_qm9(self, qm9_mols):
        """QM9 ground truth molecules should have perfect molecule stability."""
        mol_stab = qm9_mols.molecule_stability()
        assert mol_stab == 1.0, f"Expected 100% molecule stability, got {mol_stab:.2%}"


class TestValidityXyz2mol:
    """Tests for xyz2mol validity (rdDetermineBonds)."""

    def test_validity_perfect_for_qm9(self, qm9_mols):
        """QM9 ground truth molecules should all be valid."""
        val = qm9_mols.validity()
        assert val == 1.0, f"Expected 100% validity, got {val:.2%}"

    def test_keep_valid_returns_all(self, qm9_mols):
        """All QM9 molecules should be kept."""
        valid = qm9_mols.keep_valid()
        assert len(valid) == len(qm9_mols)


class TestValidityWithSmiles:
    """Tests for SMILES-based validity (pymatgen → PDB → RDKit)."""

    def test_validity_with_smiles_perfect_for_qm9(self, qm9_mols):
        """QM9 ground truth molecules should all be valid by SMILES check."""
        val = qm9_mols.validity_with_smiles()
        assert val == 1.0, f"Expected 100% SMILES validity, got {val:.2%}"

    def test_keep_valid_with_smiles_returns_all(self, qm9_mols):
        """All QM9 molecules should be kept."""
        valid = qm9_mols.keep_valid_with_smiles()
        assert len(valid) == len(qm9_mols)


class TestUniqueness:
    """Tests for uniqueness metrics."""

    def test_uniqueness_perfect_for_distinct_mols(self, qm9_mols):
        """3 distinct QM9 molecules should all be unique."""
        uniq = qm9_mols.uniqueness()
        assert uniq == 1.0, f"Expected 100% uniqueness, got {uniq:.2%}"

    def test_uniqueness_with_smiles_perfect_for_distinct_mols(self, qm9_mols):
        """3 distinct QM9 molecules should all be unique by SMILES."""
        uniq = qm9_mols.uniqueness_with_smiles()
        assert uniq == 1.0, f"Expected 100% uniqueness, got {uniq:.2%}"


class TestConsistency:
    """Tests that metrics are consistent with each other."""

    def test_all_metrics_run_without_error(self, qm9_mols):
        """Smoke test: all metrics should run without raising."""
        qm9_mols.validity()
        qm9_mols.validity_with_smiles()
        qm9_mols.uniqueness()
        qm9_mols.uniqueness_with_smiles()
        qm9_mols.atom_stability()
        qm9_mols.molecule_stability()

    def test_validity_methods_agree_on_clean_data(self, qm9_mols):
        """Both validity methods should agree on clean QM9 data."""
        val_x2m = qm9_mols.validity()
        val_smi = qm9_mols.validity_with_smiles()
        assert val_x2m == val_smi, (
            f"Validity methods disagree on clean data: "
            f"xyz2mol={val_x2m:.2%}, SMILES={val_smi:.2%}"
        )

    def test_add_bonds_does_not_change_count(self, qm9_mols):
        """Adding bonds should not change the number of molecules."""
        mols_with_bonds = qm9_mols.add_bonds()
        assert len(mols_with_bonds) == len(qm9_mols)