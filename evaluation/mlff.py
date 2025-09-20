import glob
import time
import argparse
import pandas as pd
from functools import partial
from ase import Atoms
from ase.io import read
from pathlib import Path
from tqdm import tqdm
from fairchem.core import pretrained_mlip, FAIRChemCalculator


def compute_energy(cif_file, predictor):
    sample_mof = read(cif_file)
    sample_mof.calc = FAIRChemCalculator(predictor, task_name="odac")
    energy = sample_mof.get_potential_energy()
    energy_per_atom = energy / len(sample_mof)
    return energy, energy_per_atom

def main(cif_dir, model_size):
    all_files = sorted(list((Path(cif_dir)).glob("*.cif")))

    # Create save directory
    mlff_dir = Path(cif_dir) / "mlff"
    mlff_dir.mkdir(exist_ok=True)
    print(f"INFO:: Saving MLFF properties to {mlff_dir}")

    # Calculate MLFF properties
    start_time = time.time()

    # Load model
    if model_size == "small":
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    elif model_size == "medium":
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")
    else:
        raise ValueError("Invalid model size. Choose 'small' or 'medium'.")
    
    # Compute energies
    energy_func = partial(compute_energy, predictor=predictor)
    results = [energy_func(cif_file) for cif_file in tqdm(all_files, total=len(all_files), desc="Computing energies")]
    energy, energy_per_atom = zip(*results)

    elapsed_time = time.time() - start_time
    print(f"INFO:: MLFF properties calculated in {elapsed_time} seconds")
    with open(Path(mlff_dir) / "time_mlff.txt", "w") as f:
        f.write(f"Time taken: {elapsed_time} seconds\n")

    # Save MLFF properties
    df = pd.DataFrame({"energy": energy, "energy_per_atom": energy_per_atom})
    df.to_csv(Path(mlff_dir) / "mlff.csv", index=False)
    print(df.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str)
    parser.add_argument("--model_size", type=str, default="small")
    args = parser.parse_args()
    main(cif_dir=args.cif_dir, model_size=args.model_size)