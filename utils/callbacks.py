from pytorch_lightning import Callback
from models.transformers import TransformerEncoder


def find_named_modules_by_type(module, layer_type):
    """
    Recursively finds all named submodules of a given type.
    Returns list of (full_name, module) tuples.
    """
    matches = []
    for name, child in module.named_children():
        if isinstance(child, layer_type):
            matches.append((name, child))
        else:
            for subname, submod in find_named_modules_by_type(child, layer_type):
                full_name = f"{name}.{subname}"
                matches.append((full_name, submod))
    return matches

class LMDBCleanupCallback(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_fit_end(self, trainer, pl_module):
        self.dataset.close_lmdb()

class ResetOptimizerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # Clear optimizer and lr_scheduler
        trainer.strategy._optimizers = []
        trainer.strategy._lightning_optimizers = []
        trainer.strategy.lr_scheduler_configs = []
        
        # Reconfigure optimizer and lr_scheduler
        optimizer_dict = pl_module.configure_optimizers()
        trainer.optimizers = [optimizer_dict['optimizer']]
        trainer.lr_schedulers = [optimizer_dict['lr_scheduler']]

class TensorBoardCallback(Callback):
    def __init__(self, log_freq=100, log_grad_hist=True, log_weight_hist=True,
                 log_grad_norm=True, log_weight_norm=True, log_norm_input=True):
        self.log_freq = log_freq
        self.log_grad_hist = log_grad_hist
        self.log_weight_hist = log_weight_hist
        self.log_grad_norm = log_grad_norm
        self.log_weight_norm = log_weight_norm
        self.log_norm_input = log_norm_input

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if trainer.global_step % self.log_freq != 0:
            return

        tb_writer = trainer.logger.experiment
        grad_norms = {}
        weight_norms = {}

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if self.log_grad_hist:
                    tb_writer.add_histogram(f"grad/{name}", param.grad, trainer.global_step)
                if self.log_weight_hist:
                    tb_writer.add_histogram(f"param/{name}", param.data, trainer.global_step)

                if self.log_grad_norm:
                    grad_norms[name] = param.grad.data.norm(2)
                if self.log_weight_norm:
                    weight_norms[name] = param.data.norm(2) / param.data.numel()**0.5

        if self.log_grad_norm:
            tb_writer.add_scalars("grad_norms", grad_norms, trainer.global_step)
        if self.log_weight_norm:
            tb_writer.add_scalars("weight_norms", weight_norms, trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_norm_input or trainer.global_step % self.log_freq != 0:
            return

        all_layer_norms = {}
        for full_name, layer in find_named_modules_by_type(pl_module, TransformerEncoder):
            if hasattr(layer, 'norm_inputs') and layer.norm_inputs:
                norm = layer.norm_inputs.pop(0)
                all_layer_norms[f"{full_name}"] = norm

        trainer.logger.experiment.add_scalars("norm_inputs/ffn", all_layer_norms, trainer.global_step)
