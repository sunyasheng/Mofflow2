import re
from rdkit import Chem
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')


METALS = {
    # Alkali metals
    3, 11, 19, 37, 55, 87,  # Li, Na, K, Rb, Cs, Fr
    # Alkaline earth metals
    4, 12, 20, 38, 56, 88,  # Be, Mg, Ca, Sr, Ba, Ra
    # Transition metals
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # Sc-Zn
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Y-Cd
    72, 73, 74, 75, 76, 77, 78, 79, 80,      # Hf-Hg
    104, 105, 106, 107, 108, 109, 110, 111, 112,  # Rf-Cn
    # Lanthanides
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # La-Lu
    # Actinides
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # Ac-Lr
    # Post-transition metals
    13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116  # Al, Ga, In, Sn, Tl, Pb, Bi, Po, Nh, Fl, Mc, Lv
}


def split_by_separators(smiles: str) -> list[str]:
    """
    Splits a SMILES string by <BOS>, <EOS>, and <SEP> tokens.
    For example, <EOS> and <SEP> tokens.
    """
    return re.sub(r'<BOS>|<EOS>|<SEP>', '', smiles).split()

def contains_metal(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule contains a metal atom.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in METALS:
            return True
    return False


def charge_fix(smiles: str) -> str:
    """
    Fixes the charges in a SMILES string of a MOF.
    If the molecule is unbalanced in charge, the function will fix the charges of the metal atoms.
    The charges of linkers and isolated non-metal atoms are assumed to be correct.
    """
    metals, linkers = split_by_separators(smiles)
    metals_mol = Chem.MolFromSmiles(metals)
    linkers_mol = Chem.MolFromSmiles(linkers)
    metals_total_charge = Chem.GetFormalCharge(metals_mol)
    linkers_total_charge = Chem.GetFormalCharge(linkers_mol)
    if metals_total_charge + linkers_total_charge == 0:
        return smiles
    num_metal_atoms = sum(atom.GetAtomicNum() in METALS for atom in metals_mol.GetAtoms())
    new_charge_per_metal = abs(linkers_total_charge) // num_metal_atoms
    if new_charge_per_metal * num_metal_atoms != abs(linkers_total_charge):
        warnings.warn(f"The charge of the linkers is not divisible by the number of metal atoms. The charge of the linkers is {linkers_total_charge} and the number of metal atoms is {num_metal_atoms}. The new charge per metal is {new_charge_per_metal}.")

    editable_metals_mol = Chem.RWMol(metals_mol)
    for atom in editable_metals_mol.GetAtoms():
        if atom.GetAtomicNum() in METALS:
            atom.SetFormalCharge(new_charge_per_metal)
    new_metals_mol = editable_metals_mol.GetMol()
    return "<BOS> " + Chem.MolToSmiles(new_metals_mol) + " <SEP> " + linkers + " <EOS>"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_smiles", type=str)
    args = parser.parse_args()
    print(charge_fix(args.input_smiles))