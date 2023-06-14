import os
import socket
import tempfile
from typing import Any, Callable

import torch
import torch.distributed as dist

from .argparse.parse_args import parse_args
from .utils.seed import set_seed

USE_DISTRIBUTED = "NCCL_SYNC_FILE" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ


def _setup_torchelastic():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, local_rank, world_size


def _setup_slurm():
    slurm_procid = int(os.environ["SLURM_PROCID"])
    slurm_nodeid = int(os.environ["SLURM_NODEID"])
    slurm_localid = int(os.environ["SLURM_LOCALID"])
    slurm_ntasks = int(os.environ["SLURM_NTASKS"])

    tasks_per_node = slurm_procid // slurm_nodeid if slurm_nodeid > 0 else slurm_procid

    # Calculate the local rank and world size
    local_rank = slurm_localid
    world_size = slurm_ntasks
    rank = slurm_nodeid * tasks_per_node + slurm_localid

    dist.init_process_group(
        backend="nccl",
        init_method=f'file://{os.environ["NCCL_SYNC_FILE"]}',
        world_size=world_size,
        rank=rank,
    )

    return rank, local_rank, world_size


def _ddp_setup():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError("Cannot initialize NCCL without visible CUDA devices.")

    hostname = socket.gethostname()
    print(f"Setting up DDP on {hostname}.")
    if "TORCHELASTIC_RUN_ID" in os.environ:
        print("TorchElastic detected.")
        _setup = _setup_torchelastic
    elif "NCCL_SYNC_FILE" in os.environ:
        print("Detected NCCL_SYNC_FILE. Assuming SLURM cluster.")
        _setup = _setup_slurm
    else:
        raise ValueError("Unable to detect DDP setup.")

    rank, local_rank, world_size = _setup()

    print(
        f"{hostname} ready! Rank: {rank}. Local rank: {local_rank}. World size: {world_size}."
    )
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    device = f"cuda:{int(devices[local_rank])}"
    torch.cuda.set_device(device)

    assert dist.is_initialized()

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def fire(function: Callable[[dict[Any, Any]], None]):
    config, name, experiment = parse_args()
    seed = config["seed"]

    assert isinstance(seed, int)
    seed = set_seed(seed)
    tempdir = tempfile.TemporaryDirectory()

    dist_cfg = None
    if USE_DISTRIBUTED:
        dist_cfg = _ddp_setup()
    config["dist"] = dist_cfg

    function(config)

    tempdir.cleanup()
    if dist.is_initialized():
        dist.destroy_process_group()
