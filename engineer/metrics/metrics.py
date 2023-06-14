import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")


USE_DISTRIBUTED = "RANK" in os.environ
if USE_DISTRIBUTED:
    import torch.distributed as dist


def detach_and_cast(input, device, detach=True):
    if isinstance(input, torch.Tensor):
        if device is not None:
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
        input[k] = detach_and_cast(input[k], device)
    return input


def gather(input):
    if isinstance(input, torch.Tensor):
        global_input = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(global_input, input.contiguous())
        return torch.cat(global_input)

    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = gather(input[k])
    return input


def all_gather(compute):
    def wrapper(metric):
        if not USE_DISTRIBUTED:
            return compute(metric)
        metric.collection = gather(metric.collection)
        return compute(metric)

    return wrapper


class Metric:
    def __init__(self, to_cpu=False):
        self.to_cpu = to_cpu
        self.collection = []

    def empty(self):
        return len(self.collection) == 0

    def update(self, input):
        self.collection.append(detach_and_cast(input, "cpu" if self.to_cpu else None))

    def compute(self):
        raise NotImplementedError

    def reset(self):
        self.collection.clear()


class MetricCollection:
    def __init__(self, metrics):
        self.metrics = metrics

    def empty(self):
        return all(metric.empty() for metric in self.metrics.values())

    def update(self, **kwargs):
        for name, metric in self.metrics.items():
            metric.update(kwargs[name])

    def compute(self):
        result = {}
        for name, metric in self.metrics.items():
            values = metric.compute()
            if isinstance(values, torch.Tensor):
                result[name] = values
            elif isinstance(values, dict):
                result.update(values)
            else:
                raise ValueError(f"Unknown return type {type(values)}.")

        return result

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def keys(self):
        return self.metrics.keys()

    def __repr__(self) -> str:
        return f"MetricCollection({self.metrics})"


class Accuracy(Metric):
    @all_gather
    def compute(self):
        cat = torch.cat(self.collection)
        return cat.sum(dim=0) / cat.numel()


class Loss(Metric):
    @all_gather
    def compute(self):
        return torch.mean(torch.cat(self.collection), dim=0)


def build_roc(labels, score, t_eff=[0.3, 0.5]):
    if not isinstance(t_eff, list):
        t_eff = [t_eff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in t_eff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS


class LorentzMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @all_gather
    def compute(self):
        preds, target = zip(*self.collection)
        preds = torch.cat(preds).cpu().numpy()
        target = torch.cat(target).cpu().numpy()
        assert preds.shape == target.shape
        assert preds.min() >= 0 and preds.max() <= 1

        fpr, tpr, threshold, eB, eS = build_roc(target, preds)

        auc = roc_auc_score(target, preds)

        return {"auc": auc, "eB_0.3": eB[0], "eB_0.5": eB[1]}
