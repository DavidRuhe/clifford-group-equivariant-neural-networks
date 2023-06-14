import os

import torch


class Checkpoint:
    def __init__(self, metrics=None, dir=None):
        super().__init__()

        self.dir = dir
        self._cached_model_state_dict = None
        self._cached_optimizer_state_dict = None
        self._cached_epoch = None
        self._cached_step = None

        if dir is not None:
            metrics = self.load_checkpoint(dir)

        if type(metrics) == str:
            metrics = (metrics,)
        if type(metrics) in (list, tuple):
            metrics = {m: float("inf") for m in metrics}

        self.best_metrics = metrics

        self.save_paths = {}

    def load_checkpoint(self, dir):
        state_dict = torch.load(dir)
        model = state_dict["model"]
        optimizer = state_dict["optimizer"]
        metrics = state_dict["metrics"]
        epoch = state_dict["epoch"]
        step = state_dict["step"]
        self._cached_model_state_dict = model
        self._cached_optimizer_state_dict = optimizer
        self._cached_epoch = epoch
        self._cached_step = step
        return metrics

    def restore(self, trainer, model, optimizer):
        if self._cached_model_state_dict is not None:
            if torch.distributed.is_initialized():
                model.module.load_state_dict(self._cached_model_state_dict)
            else:
                model.load_state_dict(self._cached_model_state_dict)
            print(f"Successfully restored model state dict from {self.dir}!")
        if self._cached_optimizer_state_dict is not None:
            optimizer.load_state_dict(self._cached_optimizer_state_dict)
            print(f"Successfully restored optimizer state dict from {self.dir}!")

        if self._cached_epoch is not None:
            trainer.current_epoch = self._cached_epoch
            print(f"Set current epoch to {self._cached_epoch}.")

        if self._cached_step is not None:
            trainer.global_step = self._cached_step
            print(f"Set global step to {self._cached_step}.")

        self._cached_epoch = None
        self._cached_step = None
        self._cached_model_state_dict = None
        self._cached_optimizer_state_dict = None

    @property
    def _is_master(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        else:
            return True

    def on_test_end(self, trainer, model, optimizer, metrics, *args, **kwargs):
        should_write = (
            self._is_master
            and trainer.logger is not None
            and trainer.logger.dir is not None
        )

        epoch = trainer.current_epoch
        step = trainer.global_step

        for m, v in self.best_metrics.items():
            if metrics[m] < v:
                self.best_metrics[m] = metrics[m]

                model_state_dict = (
                    model.module.state_dict()
                    if torch.distributed.is_initialized()
                    else model.state_dict()
                )
                checkpoint = {
                    "model": model_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "metrics": self.best_metrics,
                    "epoch": epoch,
                    "step": step,
                }

                if should_write:
                    alias = f"best_{m.replace('/', '_')}"
                    save_path = os.path.join(
                        trainer.logger.dir,
                        alias,
                    )

                    torch.save(checkpoint, save_path)
                    trainer.logger.save_model(save_path, alias=alias)

                    if m in self.save_paths:
                        os.remove(self.save_paths[m])
                    self.save_paths[m] = save_path

                    print_metric = (
                        metrics[m].item()
                        if type(metrics[m]) == torch.Tensor
                        else metrics[m]
                    )
                    print(
                        f"Metric {m} improved to {print_metric:.4f}, saving checkpoint. Saved checkpoint to {save_path}. Initializing test loop."
                    )
                trainer.should_test = True
