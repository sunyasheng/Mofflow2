import os
import time
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rdkit import RDLogger
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch_geometric.utils import scatter
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from utils.environment import PROJECT_ROOT
from utils.lmdb import read_lmdb, write_lmdb
from utils import conformer_matching as cu
from utils import data as du


class MOFMatcher:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        matcher_cfg = cfg.preprocess.mof_matcher

        # Task
        self.task = process_cfg.task # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.metal_dir = process_cfg.metal_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus

        # Load metal library
        with open(f"{self.metal_dir}/{self.task}/metal_lib_train.pkl", "rb") as f:
            metal_lib = pickle.load(f)
        self.metal_bb_library = metal_lib['metal_bb_library']

        # Hyperparameters
        ## Optimizer (differential_evolution)
        self.steps = matcher_cfg.optimizer.steps
        self.popsize = matcher_cfg.optimizer.popsize
        self.maxiter = matcher_cfg.optimizer.maxiter
        ## Tolerance (StructureMatcher)
        self.ltol = matcher_cfg.tolerance.ltol
        self.stol = matcher_cfg.tolerance.stol
        self.angle_tol = matcher_cfg.tolerance.angle_tol

    @staticmethod
    def _recenter_coords(coords, bb_num_vec):
        """
        Recenter coordinates so that the average of the building block centroids is at the origin.

        Args:
            coords (torch.Tensor): [num_atoms, 3], atomic coordinates.
            bb_num_vec (torch.Tensor): [num_bbs], number of atoms in each building block.
        Returns:
            torch.Tensor: [num_atoms, 3], recentered coordinates.
        """
        # Compute centroids
        bb_vec = du.repeat_interleave(bb_num_vec)
        bb_centroids = scatter(coords, bb_vec, dim=0, reduce='mean')
        bb_centroid_mean = bb_centroids.mean(dim=0, keepdim=True) # [1, 3]

        # Recenter coordinates
        recentered_coords = coords - bb_centroid_mean # [num_atoms, 3]
        return recentered_coords
    
    def process_one(self, idx, value):
        # Disable warnings within each process
        RDLogger.DisableLog('rdApp.*')

        try:
            feats = pickle.loads(value)

            # Initialize if not present
            feats.setdefault('matched_coords', [])
            feats.setdefault('rmsd', [])

            # Skip if already matched
            if feats['rmsd'] and feats['rmsd'][-1] is not None:
                return idx, value

            # Set hyperparameters
            popsize = self.popsize + 10 * self.num_prev_trial
            maxiter = self.maxiter + 10 * self.num_prev_trial

            # Perform mof matching
            matched_mols = cu.mof_matching(
                feats=feats,
                match_metal=True,
                match_organic=True,
                metal_bb_library=self.metal_bb_library,
                steps=self.steps,
                popsize=popsize,
                maxiter=maxiter
            )
            matched_coords = cu.get_matched_coords(matched_mols)

            # Compute RMSD
            gt_structure = Structure(
                lattice=feats['cell_1'],
                species=feats['atom_types'],
                coords=feats['gt_coords'],
                coords_are_cartesian=True
            )
            matched_structure = Structure(
                lattice=feats['cell_1'],
                species=feats['atom_types'],
                coords=matched_coords,
                coords_are_cartesian=True
            )
            matcher = StructureMatcher(ltol=self.ltol, stol=self.stol, angle_tol=self.angle_tol)
            rmsd = matcher.get_rms_dist(gt_structure, matched_structure)
            rmsd = rmsd if rmsd is None else rmsd[0]

            # Recenter matched coordinates
            matched_coords = self._recenter_coords(matched_coords, feats['bb_num_vec'])

            # Append results
            feats['matched_coords'].append(matched_coords)
            feats['rmsd'].append(rmsd)

            new_value = pickle.dumps(feats)
            return idx, new_value
        except Exception as e:
            print(f"Failed to process {idx}: {e}")
            return idx, None

    def process(self, split="train"):
        print(f"Matching {split} split...")

        # Start timer
        start_time = time.time()

        # Set up base directory
        base_dir = f"{self.lmdb_dir}/{self.task}"

        # Determine src file
        pattern = f"MetalOxo_matched_{split}_*.lmdb"
        self.num_prev_trial = len(list(Path(base_dir).glob(pattern)))

        if self.num_prev_trial == 0:
            print(f"No previous trials found for {split} split")
            src_env = read_lmdb(f"{base_dir}/MetalOxo_feats_{split}.lmdb")
        else:
            print(f"{self.num_prev_trial} previous trial(s) found for {split} split")
            src_env = read_lmdb(f"{base_dir}/MetalOxo_matched_{split}_{self.num_prev_trial}.lmdb")

        # Read data
        data_dict = {}
        with src_env.begin() as src_txn:
            num_src_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_src_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Process data
        feats_list = Parallel(n_jobs=self.num_cpus)(delayed(self.process_one)(idx, value) for idx, value in tqdm(data_dict.items()))

        # Write extracted features to LMDB
        dest_env = write_lmdb(f"{base_dir}/MetalOxo_matched_{split}_{self.num_prev_trial+1}.lmdb")
        with dest_env.begin(write=True) as dest_txn:
            for idx, value in tqdm(feats_list, desc="Writing data"):
                if value is not None:
                    key_bytes = f"{idx}".encode('ascii')
                    dest_txn.put(key_bytes, value)

        num_dest_entries = dest_env.stat()['entries']
        print(f"Remaining samples for trial {self.num_prev_trial+1}: {num_dest_entries}/{num_src_entries}")   
        dest_env.close()
        
        # End timer
        print(f"Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):

    mofmatcher = MOFMatcher(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        mofmatcher.process(split=split)

if __name__ == "__main__":
    main()
