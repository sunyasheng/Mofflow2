import os
import time
import random
import numpy as np
import hydra
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
from data.dataset import MOFDataset
from data.dataset_gen import MOFGenDataset
from data.dataloader import MOFDatamodule
from models.flow_module import FlowModule
from utils.environment import PROJECT_ROOT
from utils import experiment as eu


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))

class Predictor:

    def __init__(self, cfg: DictConfig):
        # Load checkpoint config
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Setup output directory
        self.inference_dir = Path(ckpt_path).parents[1] / 'inference'
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        log.info(f'Saving results to {self.inference_dir}')

        # Merge configs
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)

        # Set config
        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._data_cfg = cfg.data
        
        # Save inference config
        config_path = self.inference_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Set seed
        if self._infer_cfg.seed is not None:
            log.info(f'Setting seed to {self._infer_cfg.seed}')
            seed_everything(self._infer_cfg.seed, workers=True)
        
        log.info(f'Using device {self._infer_cfg.num_devices} devices.')

        # Load model from checkpoint
        self._module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
        )

        # Setup datamodule
        self._datamodule = self._setup_datamodule()
            
    def _setup_datamodule(self):
        if self._infer_cfg.task == 'csp':
            pred_dataset = MOFDataset(
                dataset_cfg=self._data_cfg,
                split=self._infer_cfg.csp.split,
            )
        elif self._infer_cfg.task == 'gen':
            gen_cfg = self._infer_cfg.gen
            pred_dataset = MOFGenDataset(
                metal_lib_path=gen_cfg.metal_lib_path,
                mof_seqs_path=gen_cfg.mof_seqs_path,
                sample_limit=gen_cfg.sample_limit,
            )

        datamodule = MOFDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=None,
            valid_dataset=None,
            pred_dataset=pred_dataset
        )

        return datamodule
    
    @staticmethod
    def concat_pred_list(pred_list):
        """
        Args:
            pred_list: list of predictions from the model, each element is a dictionary containing
                "cart_coords", "num_atoms", "atom_types", "lattices", and "gt_data".
        Returns:
            A dictionary containing concatenated predictions for "cart_coords", "num_atoms", "atom_types", and "lattices".
        """
        # Concatenate predictions along appropriate dimensions
        cart_coords = torch.cat([p["cart_coords"] for p in pred_list], dim=1)  # [k, total_atoms, 3]
        num_atoms = torch.cat([p["num_atoms"] for p in pred_list], dim=1)      # [k, total_graphs]
        atom_types = torch.cat([p["atom_types"] for p in pred_list], dim=1)    # [k, total_atoms]
        lattices = torch.cat([p["lattices"] for p in pred_list], dim=1)        # [k, total_graphs, 6]

        # Combine ground truth data
        gt_data_list = [data for p in pred_list for data in p["gt_data"]]
        gt_data_batch = Batch.from_data_list(gt_data_list, exclude_keys=['rotable_atom_mask'])

        return {
            "cart_coords": cart_coords,
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lattices": lattices,
            "gt_data_batch": gt_data_batch
        }
        
    def predict(self):
        # Setup trainer
        trainer = Trainer(**self._infer_cfg.trainer)

        # Predict
        start_time = time.time()
        pred_list = trainer.predict(
            model=self._module,
            datamodule=self._datamodule,
        )
        elapsed_time = time.time() - start_time
        log.info(f'Prediction time: {elapsed_time:.2f}s')

        # Concatenate per-rank predictions
        results = self.concat_pred_list(pred_list)

        # Save predictions
        save_path = os.path.join(self.inference_dir, f'predictions_{trainer.global_rank}.pt')
        torch.save(results, save_path)
        print(f'Saved rank {trainer.global_rank} predictions to {save_path}')


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="inference")
def run(cfg: DictConfig) -> None:    
    predictor = Predictor(cfg)
    predictor.predict()

if __name__ == '__main__':
    run()