import numpy as np
from torch.utils.data import DataLoader


class O5Synthetic(object):
    def __init__(self, N=1024):
        super().__init__()
        d = 5
        self.dim = 2 * d
        self.X = np.random.randn(N, self.dim)
        ri = self.X.reshape(-1, 2, 5)
        r1, r2 = ri.transpose(1, 0, 2)
        self.Y = (
            np.sin(np.sqrt((r1**2).sum(-1)))
            - 0.5 * np.sqrt((r2**2).sum(-1)) ** 3
            + (r1 * r2).sum(-1)
            / (np.sqrt((r1**2).sum(-1)) * np.sqrt((r2**2).sum(-1)))
        )
        self.Y = self.Y[..., None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)  # can add and subtract arbitrary tensors
        Xscale = (
            np.sqrt((self.X.reshape(N, 2, d) ** 2).mean((0, 2)))[:, None] + 0 * ri[0]
        ).reshape(self.dim)
        self.stats = 0, Xscale, self.Y.mean(axis=0), self.Y.std(axis=0)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


NUM_TEST = 16384


class O5Dataset:
    def __init__(self, num_samples=1024, batch_size=32):
        super().__init__()
        self.train_dataset = O5Synthetic(num_samples)
        self.val_dataset = O5Synthetic(NUM_TEST)
        self.test_dataset = O5Synthetic(NUM_TEST)

        self.batch_size = batch_size

        self.ymean, self.ystd = self.train_dataset.stats[-2].item(), self.train_dataset.stats[-1].item()

        self._normalize_datasets()

    def _normalize_datasets(self):
        Xmean, Xscale, Ymean, Ystd = self.train_dataset.stats
        self.train_dataset.X -= Xmean
        self.train_dataset.X /= Xscale
        self.train_dataset.Y -= Ymean
        self.train_dataset.Y /= Ystd

        self.val_dataset.X -= Xmean
        self.val_dataset.X /= Xscale
        self.val_dataset.Y -= Ymean
        self.val_dataset.Y /= Ystd

        self.test_dataset.X -= Xmean
        self.test_dataset.X /= Xscale
        self.test_dataset.Y -= Ymean
        self.test_dataset.Y /= Ystd

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
