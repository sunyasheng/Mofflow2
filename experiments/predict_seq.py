import os
import time
import math
import json
import hydra
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
from x_transformers import XLAutoregressiveWrapper
from omegaconf import DictConfig, OmegaConf
from models.seq_module import MOFSequenceModule
from data.tokenizer import SmilesTokenizer
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

        ##### Setup output directory #####
        temperature = cfg.inference.temperature
        dir_name = f"temp_{temperature}"

        # Append target property if conditional
        is_model_conditional = ckpt_cfg.model.get('conditional', False)
        if is_model_conditional:
            target_prop = cfg.inference.get('target_property', 'unspecified')
            dir_name += f"_target-{target_prop}"
        else:
            dir_name += "_unconditional"
        
        # Create inference directory
        self.inference_dir = Path(ckpt_path).parents[1] / 'inference' / dir_name
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        log.info(f'Saving results to {self.inference_dir}')
        ##################################

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

        # Setup tokenizer
        self._tokenizer = self._setup_tokenizer()
        
        # Load model from checkpoint
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._module = MOFSequenceModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            tokenizer=self._tokenizer,
        ).to(self._device)
        self._module.eval()

    def _setup_tokenizer(self):
        tokenizer = SmilesTokenizer()
        tokenizer.load_vocab(self._data_cfg.vocab_path)
        return tokenizer
        
    def predict(self):
        # Predict
        start_time = time.time()

        # Settings
        total_samples = self._infer_cfg.total_samples
        batch_size = self._infer_cfg.batch_size
        max_seq_len = self._cfg.model.max_seq_len
        temperature = self._infer_cfg.temperature

        # Tokens
        bos_token = self._tokenizer.vocab["<BOS>"]
        eos_token = self._tokenizer.vocab["<EOS>"]
        pad_token = self._tokenizer.vocab["<PAD>"]

        # Target property
        context = None
        if self._cfg.model.get('conditional', False):
            target = self._infer_cfg.get('target_property')
            if target is None:
                raise ValueError("Target property must be specified for conditional generation.")
            
            log.info(f"Preparing conditional generation with target property: {target}")
            target_value = torch.full((batch_size, 1), target, dtype=torch.float32, device=self._device)

            # Embed property
            context = self._module.prop_linear(target_value) + self._module.prop_fourier(target_value)
            context = context.unsqueeze(1)  # [B, 1, dim]

        log.info(f'Predicting {self._infer_cfg.total_samples} samples...')
        all_sequences = []
        with torch.no_grad():
            num_batches = math.ceil(total_samples / batch_size)
            for _ in tqdm(range(num_batches), desc="Generating sequences"):
                prompt = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=self._device)

                xl_wrapper = XLAutoregressiveWrapper(
                    self._module.model,
                    ignore_index=-100,
                    pad_value=pad_token,
                )

                generated = xl_wrapper.generate(
                    prompt,
                    mask=None,
                    seq_len=max_seq_len,
                    eos_token=eos_token,
                    temperature=temperature,
                    context=context,
                )

                decoded = ["".join(self._tokenizer.decode(seq.tolist())) for seq in generated]
                all_sequences.extend(decoded)

        elapsed_time = time.time() - start_time
        log.info(f'Prediction time: {elapsed_time:.2f}s')

        # Truncate
        all_sequences = all_sequences[:total_samples]

        # Save predictions
        # Include target_property in filename for conditional models
        if self._cfg.model.get('conditional', False):
            target_prop = self._infer_cfg.get('target_property', 'unspecified')
            save_path = self.inference_dir / f"preds_samples-{self._infer_cfg.total_samples}_target-{target_prop}.json"
        else:
            save_path = self.inference_dir / f"preds_samples-{self._infer_cfg.total_samples}.json"
        save_path.write_text(json.dumps(all_sequences, indent=2))
        log.info(f"Saved predictions to {save_path}")

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="inference_seq")
def run(cfg: DictConfig) -> None:    
    predictor = Predictor(cfg)
    predictor.predict()

if __name__ == '__main__':
    run()