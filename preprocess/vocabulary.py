import os
import time
import json
import hydra
import pickle
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from openbabel import pybel
from collections import OrderedDict, Counter
from omegaconf import DictConfig
from utils import molecule as mu
from utils.lmdb import read_lmdb
from utils.environment import PROJECT_ROOT

RDLogger.DisableLog('rdApp.*')
pybel.ob.obErrorLog.SetOutputLevel(0)
warnings.filterwarnings("ignore")

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "<SEP>": 4,
    "<CLS>": 5,
    "<MASK>": 6,
}

def process_one(value):
    try:
        feats = pickle.loads(value)
        bb_mols = feats['bb_mols']
        results = []

        for mol_str in bb_mols:
            mol = Chem.MolFromMolBlock(mol_str, removeHs=True) # NOTE: remove hydrogens
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            type_id = 0 if mu.is_metal_bb(mol) else 1
            results.append((smiles, type_id))

        return results
    except Exception:
        return []

class ExtractVocabulary:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        self.lmdb_dir = process_cfg.lmdb_dir
        self.vocab_dir = process_cfg.vocab_dir
        if not os.path.exists(self.vocab_dir):
            os.makedirs(self.vocab_dir)
        self.num_cpus = process_cfg.num_cpus  

    def process(self, split="train"):
        print(f"Extracting vocabulary from {split} split...")
        start_time = time.time()

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/MetalOxo_absolute_torsion_{split}.lmdb")
        with src_env.begin() as src_txn:
            num_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Extract (smiles, type_id) pairs from all MOFs
        all_results = Parallel(n_jobs=self.num_cpus)(
            delayed(process_one)(value) for value in tqdm(data_dict.values(), desc="Processing data")
        )

        # Deduplicate
        smiles_to_bb_type = OrderedDict()
        smiles_counter = Counter()

        for result in all_results:
            for smiles, type_id in result:
                smiles_to_bb_type.setdefault(smiles, type_id)
                smiles_counter.update([smiles])

        # Start vocab with special tokens
        smiles_to_index = {token: idx for token, idx in SPECIAL_TOKENS.items()}
        smiles_to_bb_type.update({token: 2 for token in SPECIAL_TOKENS if token not in smiles_to_bb_type})

        # Assign indices to remaining tokens
        offset = len(SPECIAL_TOKENS)
        for idx, (smi, _) in enumerate(smiles_to_bb_type.items()):
            if smi not in smiles_to_index:
                smiles_to_index[smi] = idx + offset

        index_to_smiles = {idx: smi for smi, idx in smiles_to_index.items()}

        vocab = {
            "smiles_to_index": smiles_to_index,
            "smiles_to_bb_type": smiles_to_bb_type,
            "index_to_smiles": index_to_smiles,
            "smiles_counter": dict(smiles_counter),
        }

        # Save vocabulary
        output_file = f"{self.vocab_dir}/vocab_{split}.json"
        with open(output_file, "w") as f:
            json.dump(vocab, f, indent=2)

        print(f"INFO:: Saved vocabulary ({len(smiles_to_index)} entries) to {output_file}")
        print(f"INFO::Time taken: {time.time() - start_time:.2f} seconds")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    extractor = ExtractVocabulary(cfg=cfg)
    extractor.process(split="train")

if __name__ == "__main__":
    main()
