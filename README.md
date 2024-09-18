# molmetrics: Metrics for 3D Molecular Structures

`molmetrics` is a collection of metrics for 3D molecular structures, built on top of the RDKit, OpenBabel, Posebusters and EDM libraries.


## Installation

```bash
pip install git+https://github.com/atomicarchitects/molmetrics
```

## Usage
```python
import molmetrics as mm

# Load molecules from a directory.
mols = mm.RDKitMolecules.from_directory('../tests/data/qm9')

# Add bonds to the molecules.
mols = mols.add_bonds()

# Check the validity and uniqueness of the molecules.
print(f"Validity: {mols.validity()}")
print(f"Uniqueness: {mols.uniqueness()}")

# Calculate the bond length distribution.
bond_lengths = mols.bond_lengths()
```

See the `examples` folder for more usage!