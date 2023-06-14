import torch
import torch.optim
from torch.nn.parallel import DistributedDataParallel

import engineer
from engineer.schedulers.cosine import CosineAnnealingLR


def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    model_config = config["model"]
    model_module = engineer.load_module(model_config.pop("module"))
    model = model_module(**model_config)

    if config["dist"] is not None:
        local_rank = config["dist"]["local_rank"]
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    print(f"Using device: {device}")

    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )
    steps = config["trainer"]["max_steps"]
    scheduler = config["trainer"]["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        steps,
        warmup_steps=int(1 / 64 * steps),
        decay_steps=int(1 / 4 * steps),
    )
    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config["scheduler"] = scheduler
    trainer = trainer_module(
        **trainer_config,
    )
    trainer.fit(model, optimizer, train_loader, val_loader, test_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    engineer.fire(main)
