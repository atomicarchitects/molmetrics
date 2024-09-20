# molmetrics: Metrics for 3D Molecular Structures
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

    
`molmetrics` is a collection of metrics for 3D molecular structures, built on top of the [RDKit](https://www.rdkit.org/), [Open Babel](https://openbabel.org/index.html), [Posebusters](https://github.com/maabuu/posebusters) and [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules) libraries.


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

## Citing

This repository would not be possible without the effort of many others. Please cite the following papers and libraries if you use `molmetrics`:

```bibtex
@software{rdkit,
	title        = {{rdkit/rdkit: 2024\_09\_1 (Q3 2024) Release Beta}},
	author       = {Greg Landrum and Paolo Tosco and Brian Kelley and Ricardo Rodriguez and David Cosgrove and Riccardo Vianello and sriniker and Peter Gedeck and Gareth Jones and NadineSchneider and Eisuke Kawashima and Dan Nealschneider and Andrew Dalke and Matt Swain and Brian Cole and Samo Turk and Aleksandr Savelev and Alain Vaucher and Maciej W{\'o}jcikowski and Ichiru Take and Vincent F. Scalfani and Rachel Walker and Daniel Probst and Kazuya Ujihara and Axel Pahl and guillaume godin and Juuso Lehtivarjo and tadhurst-cdd and Fran{\c c}ois B{\'e}renger and Jonathan Bisson},
	year         = 2024,
	month        = sep,
	publisher    = {Zenodo},
	doi          = {10.5281/zenodo.13820100},
	url          = {https://doi.org/10.5281/zenodo.13820100},
	version      = {Release\_2024\_09\_1b1},
	bdsk-url-1   = {https://doi.org/10.5281/zenodo.13820100}
}
@misc{openbabel,
	title        = {{Open Babel: An open chemical toolbox}},
	author       = {Michael Banck and Craig A. Morley and Tim Vandermeersch and Geoffrey R. Hutchison},
	year         = 2011,
	url          = {https://doi.org/10.1186/1758-2946-3-33},
	eprint       = {PMC3198950},
	eprinttype   = {pmcid}
}
@article{posebusters,
	title        = {{{{PoseBusters}}: {{AI-based}} Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences}},
	shorttitle   = {{PoseBusters}},
	author       = {Buttenschoen, Martin and Morris, Garrett M. and Deane, Charlotte M.},
	year         = 2023,
	publisher    = {The Royal Society of Chemistry},
	doi          = {10.1039/D3SC04185A},
	url          = {http://dx.doi.org/10.1039/D3SC04185A}
}
@inproceedings{edm,
	title        = {{Equivariant Diffusion for Molecule Generation in 3D}},
	author       = {Hoogeboom, Emiel and Satorras, V{\i}ctor Garcia and Vignac, Cl{\'e}ment and Welling, Max},
	year         = 2022,
	booktitle    = {International conference on machine learning},
	pages        = {8867--8887},
	organization = {PMLR}
}
@inproceedings{symphony,
	title        = {{Symphony: Symmetry-Equivariant Point-Centered Spherical Harmonics for 3D Molecule Generation}},
	author       = {Ameya Daigavane and Song Eun Kim and Mario Geiger and Tess Smidt},
	year         = 2024,
	booktitle    = {The Twelfth International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=MIEnYtlGyv}
}
```


