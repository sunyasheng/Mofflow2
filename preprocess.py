import argparse
import subprocess
import sys

# Preprocessing steps and their comments, as in the README
PREPROCESSING_STEPS = [
    {
        'comment': '''
        Step 1: Filter MOFs with > 20 bbs and > 200 atoms/bb (following MOFDiff).
        File: MetalOxo.lmdb -> MetalOxo_filtered_{split}.lmdb
        ''',
        'command': ['python', 'preprocess/filter.py'],
        'summary': 'Step 1: Filter MOFs with > 20 bbs and > 200 atoms/bb.'
    },
    {
        'comment': '''
            Step 2: Extract key features, including:
            - Cartesian coordinates
            - Atom types and features
            - Niggli-reduced unit cells
            - Symmetrically equivalent coordinates
            - Rotatable bond information
            Also verifies chemical validity of each building block using RDKit.
            File: MetalOxo_filtered_{split}.lmdb -> MetalOxo_feats_{split}.lmdb
        ''',
        'command': ['python', 'preprocess/extract_feats.py'],
        'summary': 'Step 2: Extract key features and verify chemical validity.'
    },
    {
        'comment': '''
        Step 3: Construct a metal building block library by aggregating atomic coordinates.
        This script outputs three dictionaries:
        - 'metal_mol_dict':
            key   = canonical SMILES string of a metal-containing building block
            value = list of Chem.rdchem.Mol objects found in the dataset for that SMILES
        - 'metal_bb_library':
            key   = canonical SMILES string
            value = Chem.rdchem.Mol object with coordinates averaged from list of 'metal_mol_dict'
        - 'rmsd_dict':
            key   = canonical SMILES string
            value = list of RMSD values computed between each coordinate set in metal_mol_dict and a randomly chosen reference
        File: MetalOxo_feats_{split}.lmdb -> metal_lib_{split}.pkl
        ''',
        'command': ['python', 'preprocess/metals.py'],
        'summary': 'Step 3: Construct metal building block library.'
    },
    {
        'comment': '''
        Step 4: Perform MOF matching by reconstructing full structures using:
        - Metal building block coordinates from 'metal_bb_library'
        - Organic building block coordinates generated in two ways:
            (1) Directly from RDKit
            (2) From RDKit with optimized torsion angles
        Each trial increases the resources used for torsion angle optimization, aiming to improve the number of matched coordinates.
        The matched coordinates are stored in the 'matched_coords' field as a list of coordinate arrays;
        the final element in this list corresponds to the best-matched coordinates.
        File: MetalOxo_feats_{split}.py -> MetalOxo_matched_{split}_{num_trial}.lmdb
        ''',
        'command': ['python', 'preprocess/mof_matching.py'],
        'repeat': 3,  # This will be overwritten by the CLI argument
        'summary': 'Step 4: Perform MOF matching with torsion angle optimization.'
    },
    {
        'comment': '''
        Step 5: Validity check with MOFChecker
        File: MetalOxo_matched_{split}_{num_trial}.lmdb -> MetalOxo_final_{split}.lmdb
        ''',
        'command': ['python', 'preprocess/check_mof_validity.py'],
        'summary': 'Step 5: Check validity with MOFChecker.'
    },
    {
        'comment': '''
        Convert format (for baselines):
        - python preprocess/write_csv.py # DiffCSP, FlowMM, etc.
        ''',
        'command': ['python', 'preprocess/write_csv.py'],
        'optional': True,
        'summary': '(Optional) Convert format for baselines (CSV).'
    },
    {
        'comment': '''
        Convert format (for baselines):
        - python preprocess/write_pkl.py # MOFFlow-1
        ''',
        'command': ['python', 'preprocess/write_pkl.py'],
        'optional': True,
        'summary': '(Optional) Convert format for baselines (PKL).'
    }
]

def main():
    parser = argparse.ArgumentParser(description='Run MOFFlow preprocessing pipeline.')
    parser.add_argument('--mof-matching-repeat', type=int, default=3, help='Number of times to repeat the MOF matching step (default: 3)')
    parser.add_argument('--run-conversion', action='store_true', help='Run the optional baseline format conversion steps.')
    args = parser.parse_args()

    print("Starting preprocessing pipeline...\n")

    # Update the repeat count for the MOF matching step (Step 4)
    for step in PREPROCESSING_STEPS:
        if 'mof_matching.py' in step['command']:
            step['repeat'] = args.mof_matching_repeat

    for step in PREPROCESSING_STEPS:
        # Skip baseline conversion by default
        if step.get('optional') and not args.run_conversion:
            continue
        repeat = step.get('repeat', 1)
        for i in range(repeat):
            print(f"# {step['summary']}")
            try:
                subprocess.run(step['command'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {' '.join(step['command'])}", file=sys.stderr)
                sys.exit(e.returncode)
    print("\nPreprocessing completed.")

if __name__ == '__main__':
    main() 