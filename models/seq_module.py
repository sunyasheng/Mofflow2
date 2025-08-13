import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from x_transformers import TransformerWrapper, Decoder, XLAutoregressiveWrapper
from utils.model import LinearWarmupScheduler


class MOFSequenceModule(LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        # Configs
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self.tokenizer = tokenizer

        attn_cfg = self._model_cfg.attention
        self.model = TransformerWrapper(
            num_tokens=tokenizer.vocab_size,
            max_seq_len=self._model_cfg.max_seq_len,
            attn_layers=Decoder(
                dim=attn_cfg.dim,
                depth=attn_cfg.depth,
                heads=attn_cfg.heads,
                rotary_pos_emb=attn_cfg.rotary_pos_emb,
                attn_flash=attn_cfg.attn_flash,
                use_scalenorm=attn_cfg.use_scalenorm,
            ),
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.pad_token_id = tokenizer.vocab['<PAD>']

    def _log_generated_sequence(self, tag="seq"):
        """Generate and log a single sequence from <BOS>."""
        if self.logger is None and not self.trainer.is_global_zero:
            return

        bos_token = self.tokenizer.vocab["<BOS>"]
        prompt = torch.full((1, 1), bos_token, dtype=torch.long, device=self.device)

        xl_wrapper = XLAutoregressiveWrapper(
            self.model,
            ignore_index=-100,
            pad_value=self.pad_token_id
        )
        generated = xl_wrapper.generate(
            prompt,
            mask=None,
            seq_len=self._model_cfg.max_seq_len,
            eos_token=self.tokenizer.vocab["<EOS>"],
        )

        decoded = self.tokenizer.decode(generated[0].tolist())
        decoded_str = "".join(decoded)
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_text(tag, decoded_str, self.global_step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_text(key=tag, columns=["generated"], data=[[decoded_str]], step=self.global_step)
        else:
            print(f"Global step {self.global_step}: {decoded_str}")
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, mask=attention_mask)

    def training_step(self, batch, batch_idx):
        step_start_time = time.time()

        # Compute loss
        input_ids = batch["input_ids"]       # [B, T]
        target_ids = batch["target_ids"]     # [B, T]
        attention_mask = batch["attention_mask"]  # [B, T]

        logits = self(input_ids, attention_mask=attention_mask)  # [B, T, vocab_size]
        train_loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        ####### Loggings #######
        batch_size = input_ids.shape[0]

        # Batch size
        self.log('train/batch_size', float(batch_size))

        # Loss        
        self.log("train/loss", train_loss, batch_size=batch_size)

        # Sample sequence
        sample_freq = self._exp_cfg.sample_seq_freq
        if sample_freq and self.global_step % sample_freq == 0:
            self._log_generated_sequence(tag="sample_seq")
        
        # Time
        step_time = time.time() - step_start_time
        self.log('train/examples_per_second', batch_size / step_time)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # Compute loss
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask=attention_mask)
        val_loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        ####### Loggings #######
        batch_size = input_ids.shape[0]
        self.log("valid/loss", val_loss, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self._exp_cfg.optimizer
        )

        scheduler_cfg = self._exp_cfg.lr_scheduler
        scheduler_type = self._exp_cfg.lr_scheduler_type

        if scheduler_type == 'linear_warmup':
            scheduler = LinearWarmupScheduler(optimizer, **scheduler_cfg.linear_warmup)
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_cfg.reduce_on_plateau
            )
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid/loss'
        }
