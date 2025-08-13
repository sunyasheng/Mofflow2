import os
import re
import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed


def normalize_sequence(seq: str) -> str:
    """
    Removes <BOS>, <EOS>, <PAD> tokens and whitespaces from the sequence.
    """
    return (
        seq.replace("<BOS>", "")
           .replace("<EOS>", "")
           .replace("<PAD>", "")
           .replace(" ", "")
           .strip()
    )

def extract_bbs(cleaned_str):
    metal_part, organic_part = cleaned_str.split("<SEP>", maxsplit=1)
    metal_bbs = [s for s in metal_part.split(".") if s]
    organic_bbs = [s for s in organic_part.split(".") if s]

    return metal_bbs, organic_bbs

def insert_dot(metal_bb_str):
    bracketed_atom_regex = re.compile(r"\[[^\]]+\]")
    return ".".join(bracketed_atom_regex.findall(metal_bb_str))

def validate_and_reorder(seq, metal_lib):
    """
    Validates the sequence and reorders organic building blocks by MolWt.

    Args:
        seq (str): normalized generated sequence (no <BOS>/<EOS>/<PAD> or whitespace)
        metal_lib (set[str]): set of valid metal building block SMILES

    Returns:
        is_valid (bool): True if all BBs are valid
        reordered_seq (str or None): reordered sequence if valid, else None
    """
    if "<SEP>" not in seq:
        return False, None

    metal_bbs, organic_bbs = extract_bbs(seq)

    # Validate metal building blocks
    if not all(metal in metal_lib for metal in metal_bbs):
        return False, None
    
    # Validate organic building blocks
    organic_mols = [Chem.MolFromSmiles(organic) for organic in organic_bbs]
    if any(organic is None for organic in organic_mols):
        return False, None

    # Sort by MolWt
    metal_mols = [Chem.MolFromSmiles(insert_dot(metal)) for metal in metal_bbs]
    metal_mols = sorted(metal_mols, key=lambda x: Descriptors.ExactMolWt(x))
    organic_mols = sorted(organic_mols, key=lambda x: Descriptors.ExactMolWt(x))

    # Canonicalize
    metal_smiles = [
        Chem.MolToSmiles(mol, canonical=True).replace(".", "") # Remove dots
        for mol in metal_mols
    ]
    organic_smiles = [
        Chem.MolToSmiles(mol, canonical=True)
        for mol in organic_mols
    ]
    metal_str = ".".join(metal_smiles)
    organic_str = ".".join(organic_smiles)
    reordered = f"{metal_str}<SEP>{organic_str}"
    return True, reordered

def is_novel(sequence, train_set):
    """
    Checks if the sequence is not in the training set.
    """
    return sequence not in train_set

def has_novel_bb(sequence, metal_lib, organic_lib):
    """
    Checks if the sequence contains novel building blocks.
    """
    metal_bbs, organic_bbs = extract_bbs(sequence)

    metal_novel = any(bb not in metal_lib for bb in metal_bbs)
    organic_novel = any(bb not in organic_lib for bb in organic_bbs)
    return metal_novel or organic_novel

def create_bb_lib(bb_lib_dir, train_seqs):
    """
    Create building block libraries (metals.pkl and organics.pkl) from the training sequences.
    """
    # Normalize training sequences
    train_seq = [normalize_sequence(seq) for seq in train_seqs]

    # Extract building blocks
    metal_lib, organic_lib = set(), set()
    for seq in train_seq:
        metal, organic = extract_bbs(seq)
        metal_lib.update(metal)
        organic_lib.update(organic)

    # Save building block libraries
    print(f"INFO:: Found {len(metal_lib)} unique metal building blocks.")
    print(f"INFO:: Found {len(organic_lib)} unique organic building blocks.")
    os.makedirs(bb_lib_dir, exist_ok=True)
    with open(f"{bb_lib_dir}/metals.pkl", 'wb') as f:
        pickle.dump(metal_lib, f)
    with open(f"{bb_lib_dir}/organics.pkl", 'wb') as f:
        pickle.dump(organic_lib, f)

def main(args):
    # Load generated sequences
    with open(args.generated_json, 'r') as f:
        generated_seqs = json.load(f)

    # Load training set
    with open(args.train_json, 'r') as f:
        train_seqs = json.load(f)
        train_seqs = list(train_seqs.values())

    # Load building block library
    # Create building block library if it doesn't exist
    if not os.path.exists(f"{args.bb_lib_dir}/metals.pkl") or not os.path.exists(f"{args.bb_lib_dir}/organics.pkl"):
        print(f"INFO:: Creating building block library in {args.bb_lib_dir}")
        create_bb_lib(args.bb_lib_dir, train_seqs)
    with open(f"{args.bb_lib_dir}/metals.pkl", 'rb') as f:
        metal_lib = pickle.load(f)
        assert isinstance(metal_lib, set), "Metal library should be a set of SMILES strings"
    with open(f"{args.bb_lib_dir}/organics.pkl", 'rb') as f:
        organic_lib = pickle.load(f)
        assert isinstance(organic_lib, set), "Organic library should be a set of SMILES strings"

    # Normalize sequences (remove <BOS>, <EOS>, <PAD>)
    generated_seqs = [normalize_sequence(seq) for seq in generated_seqs]
    train_seqs = [normalize_sequence(seq) for seq in train_seqs]
    train_set = set(train_seqs)
    
    # Evaluate validity, novelty, and uniqueness
    seen = set()
    results = []

    for seq in tqdm(generated_seqs, desc="Evaluating sequences"):
        valid, ordered_seq = validate_and_reorder(seq, metal_lib)

        if valid:
            novelty = is_novel(ordered_seq, train_set)
            uniqueness = ordered_seq not in seen
            seen.add(ordered_seq)
            novel_bb = has_novel_bb(ordered_seq, metal_lib, organic_lib)
        else:
            novelty = False
            uniqueness = False
            novel_bb = False

        results.append({
            "valid_smi": valid,
            "novelty": novelty,
            "uniqueness": uniqueness,
            "vnu": valid and novelty and uniqueness,
            "novel_bb": novel_bb,
        })
    
    df = pd.DataFrame(results)

    # Print summary statistics
    print(f"Valid: {df['valid_smi'].sum()}/{len(df)}")
    print(f"Novel: {df['novelty'].sum()}/{len(df)}")
    print(f"Unique: {df['uniqueness'].sum()}/{len(df)}")
    print(f"VNU: {df['vnu'].sum()}/{len(df)}")
    print(f"Novel BB: {df['novel_bb'].sum()}/{len(df)}")
    
    # Save outputs
    save_dir = Path(args.generated_json).parent
    df.to_csv(save_dir / "vnu_seq_only.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_json", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--bb_lib_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)