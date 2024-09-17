from openbabel import openbabel
from openbabel import pybel


def check_molecule_validity(
    mol: "openbabel.OBMol",
) -> bool:
    if mol.NumBonds() == 0:
        return False

    # Table of valences for each atom type.
    expected_valences = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
    }

    invalid = False
    for atom in openbabel.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        atomic_symbol = openbabel.GetSymbol(atomic_num)
        atom_valency = atom.GetExplicitValence()
        if atom_valency != expected_valences[atomic_symbol]:
            invalid = True
            break

    return not invalid
