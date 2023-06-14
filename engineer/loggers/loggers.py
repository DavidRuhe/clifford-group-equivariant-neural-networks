import torch


def _pp(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            _pp(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


class ConsoleLogger:
    def __init__(self) -> None:
        self.metrics = []
        self.dir = None

    def _log(self, dict, step):
        # Print metrics
        print()
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            print(f"{k}: {v:.4f}")
        print()

    def log_metrics(self, metrics, step):
        for m in metrics:
            if m not in self.metrics:
                print(f"Defined metric {m}.")
                self.metrics.append(m)

        return self._log(metrics, step)
