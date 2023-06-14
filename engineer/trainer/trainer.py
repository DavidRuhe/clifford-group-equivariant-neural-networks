import datetime
import os
import subprocess
import time
import warnings
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from ..callbacks.checkpoint import Checkpoint
from ..loggers.loggers import ConsoleLogger


def human_format(num: float):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(input, device, detach=True):
    if isinstance(input, torch.Tensor):
        input = input.to(device)
        if detach:
            input = input.detach()
        return input
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = to_device(input[k], device)
    return input


def run_bash_command(command: str) -> str:
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )

    if result.returncode == 0:
        output = result.stdout.strip()
        return output
    else:
        error = result.stderr.strip()
        raise RuntimeError(f"Error executing command: {error}")


def parse_time_components(time_string: str):
    days, hours, minutes, seconds = 0, 0, 0, 0

    # Splitting days if present.
    if "-" in time_string:
        try:
            days_str, time_string = time_string.split("-")
        except:
            raise ValueError(f"Invalid time format {time_string}.")
        days = int(days_str)

    # Splitting hours, minutes, and seconds.
    time_components = time_string.split(":")
    num_components = len(time_components)

    if num_components == 3:
        hours, minutes, seconds = map(int, time_components)
    elif num_components == 2:
        minutes, seconds = map(int, time_components)
    elif num_components == 1:
        seconds = int(time_components[0])
    else:
        raise ValueError(f"Invalid time format {time_string}.")

    return days, hours, minutes, seconds


def parse_slurm_time(time_string) -> datetime.timedelta:
    days, hours, minutes, seconds = parse_time_components(time_string)
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _parse_max_time(time):
    if time is None:
        return

    if time is None and "SLURM_JOB_ID" in os.environ:
        time = run_bash_command(
            "squeue -j $SLURM_JOB_ID -h --Format TimeLimit"
        ).splitlines()
        if len(time) > 1:
            warnings.warn(
                "More than one job found (array job?). Using the first one for setting the time limit."
            )
        time = time[0]

    max_time = parse_slurm_time(time)
    return max_time


class Trainer:
    def __init__(
        self,
        scheduler=None,
        logger: Any = None,
        max_steps: int = 0,
        max_time: str = None,
        limit_val_batches: int = float("inf"),
        val_check_interval: int = 1024,
        print_interval: int = 32,
        fast_dev_run: bool = False,
        wandb=None,
        callbacks=list(),
        log_interval=256,
        checkpoint=None,
        test_only=False,
    ):
        self.callbacks = callbacks

        if logger is None:
            if wandb:
                logger = WANDBLogger()
            else:
                logger = ConsoleLogger()

        if any(isinstance(c, Checkpoint) for c in callbacks):
            assert (
                checkpoint is None
            ), f"Checkpoint {checkpoint} is already in callbacks."
            checkpoint = next(c for c in callbacks if isinstance(c, Checkpoint))
        elif checkpoint is None and not any(
            isinstance(c, Checkpoint) for c in callbacks
        ):
            checkpoint = Checkpoint("val/loss")
            callbacks.append(checkpoint)
        elif isinstance(checkpoint, str):
            checkpoint = Checkpoint(dir=checkpoint)
        else:
            raise ValueError(f"Unknown checkpoint: {checkpoint}.")

        if fast_dev_run:
            print("This is a development run. Limiting the number of batches to 1.")
            max_steps = 1
            limit_val_batches = 1

        self.starting_time = datetime.datetime.now()
        self.max_time = _parse_max_time(max_time)
        self.checkpoint = checkpoint
        self.fast_dev_run = fast_dev_run
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.limit_val_batches = limit_val_batches
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.print_interval = print_interval
        self.log_interval = log_interval
        self.test_only = test_only
        self.is_distributed = dist.is_initialized()

        self.global_step = 0
        self.current_epoch = 0

        self.should_raise = None
        self.should_test = False
        self.device = None

    def _add_prefix(
        self, metrics: dict[str, torch.Tensor], prefix: str
    ) -> dict[str, torch.Tensor]:
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def train_step(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, batch: Any
    ):
        model.train()

        batch = to_device(batch, self.device)

        loss, outputs = model(batch, self.global_step)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            self.should_raise = ValueError("Loss is NaN.")

        if self.is_distributed:
            model.module.train_metrics.update(**outputs)
        else:
            model.train_metrics.update(**outputs)

        if self.global_step % self.print_interval == 0:
            print(f"Step: {self.global_step} (Training) Loss: {loss:.4f}")

    @torch.no_grad()
    def test_loop(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        test_loader: DataLoader,
        validation=False,
    ):
        model.eval()

        num_iterations = int(min(len(test_loader), self.limit_val_batches))
        t0 = time.time()

        if self.is_distributed:
            assert model.module.test_metrics.empty()
        else:
            assert model.test_metrics.empty()
        if validation:
            print_str = "Validation"
            prefix = "val"
        else:
            print_str = "Testing"
            prefix = "test"

        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= self.limit_val_batches:
                break

            batch = to_device(batch, self.device)
            _, outputs = model(batch, batch_idx)

            if self.is_distributed:
                model.module.test_metrics.update(**outputs)
            else:
                model.test_metrics.update(**outputs)

            if batch_idx % self.print_interval == 0:
                print(
                    f"Step: {self.global_step} ({print_str}) Batch: {batch_idx} / {num_iterations}"
                )

        t1 = time.time()
        s_it = (t1 - t0) / num_iterations

        if self.is_distributed:
            metrics = model.module.test_metrics.compute()
            model.module.test_metrics.reset()
        else:
            metrics = model.test_metrics.compute()
            model.test_metrics.reset()
        metrics[f"s_it"] = s_it

        metrics = self._add_prefix(metrics, prefix)

        if self.logger:
            self.logger.log_metrics(metrics, step=self.global_step)

        if validation:
            for callback in self.callbacks:
                callback.on_test_end(self, model, optimizer, metrics)

    @property
    def should_stop(self):
        if (
            self.max_time is not None
            and self.max_time < datetime.datetime.now() - self.starting_time
        ):
            print("Stopping due to max_time.")
            return True
        if self.max_steps is not None and self.global_step >= self.max_steps:
            print("Stopping due to max_steps.")
            return True
        return False

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader=None,
        test_loader=None,
    ):
        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device
        self.device = device

        if torch.cuda.is_available() and not device.type == "cuda":
            warnings.warn("CUDA is available but not being used.")

        print("\nModel Summary\n---")
        print(model)
        print(f"Total parameters: {human_format(count_parameters(model))}\n")

        if self.checkpoint:
            self.checkpoint.restore(self, model, optimizer)

        if self.test_only:
            print(f"Testing mode.")
            with torch.no_grad():
                self.test_loop(model, optimizer, test_loader, validation=False)
            return

        t0 = time.time()

        last_global_step = self.global_step

        while not self.should_stop:
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.current_epoch)
            for batch in train_loader:
                self.train_step(model, optimizer, batch)

                if self.scheduler is not None:
                    self.scheduler.step()

                lr = optimizer.param_groups[0]["lr"]

                if self.global_step % self.log_interval == 0:
                    t1 = time.time()
                    if self.is_distributed:
                        train_metrics = model.module.train_metrics.compute()
                        model.module.train_metrics.reset()
                    else:
                        train_metrics = model.train_metrics.compute()
                        model.train_metrics.reset()

                    s_it = (t1 - t0) / (self.global_step + 1 - last_global_step)
                    train_metrics["s_it"] = s_it
                    train_metrics["lr"] = lr
                    train_metrics["epoch"] = self.current_epoch

                    if self.logger:
                        train_metrics = self._add_prefix(train_metrics, "train")
                        self.logger.log_metrics(train_metrics, step=self.global_step)

                    t0 = time.time()
                    last_global_step = self.global_step

                if self.global_step % self.val_check_interval == 0:
                    if val_loader is not None and self.limit_val_batches > 0:
                        with torch.no_grad():
                            self.test_loop(
                                model, optimizer, val_loader, validation=True
                            )

                    t0 = time.time()
                    last_global_step = self.global_step

                    if self.should_test:
                        if test_loader is not None:
                            with torch.no_grad():
                                self.test_loop(
                                    model, optimizer, test_loader, validation=False
                                )
                                self.should_test = False

                self.global_step += 1

                if self.should_raise is not None:
                    raise self.should_raise

                if self.should_stop:
                    break

            self.current_epoch += 1
