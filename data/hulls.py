import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

from .o3 import MemmapDataset


class ConvexHullDataset:
    def __init__(self, num_samples=256, batch_size=32) -> None:
        super().__init__()

        dataroot = os.path.join(os.environ["DATAROOT"], "hulls")

        self.train_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_train_input.npy"),
            os.path.join(dataroot, "hulls_train_target.npy"),
            num_samples=num_samples,
        )

        self.val_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_val_input.npy"),
            os.path.join(dataroot, "hulls_val_target.npy"),
        )

        self.test_dataset = MemmapDataset(
            os.path.join(dataroot, "hulls_test_input.npy"),
            os.path.join(dataroot, "hulls_test_target.npy"),
        )

        self.batch_size = batch_size

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


def signed_volume_of_simplex(simplex):
    matrix = np.column_stack(simplex)
    return np.linalg.det(matrix) / np.math.factorial(matrix.shape[0])


def get_signed_hull_volume(hull):
    volume = 0
    ref_point = hull.points[0]
    for simplex_indices in hull.simplices:
        simplex_points = [hull.points[index] - ref_point for index in simplex_indices]
        volume += signed_volume_of_simplex(simplex_points)
    return volume


if __name__ == "__main__":
    dataroot = os.path.join(os.environ["DATAROOT"], "hulls")

    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    from scipy.spatial import ConvexHull

    def _generate_meshes(n_points):
        meshes = []
        for i in trange(n_points):
            points = np.random.randn(16, 5)
            hull = ConvexHull(points)
            volume = hull.volume 
            meshes.append((points, volume))

        meshes, volumes = zip(*meshes)
        meshes = np.stack(meshes).astype(np.float32)
        volumes = np.stack(volumes).astype(np.float32)
        return meshes, volumes

    train_meshes, train_volumes = _generate_meshes(16384)
    val_meshes, val_volumes = _generate_meshes(16384)
    test_meshes, test_volumes = _generate_meshes(16384)
    np.save(os.path.join(dataroot, "hulls_train_input.npy"), train_meshes)
    np.save(os.path.join(dataroot, "hulls_train_target.npy"), train_volumes)
    np.save(os.path.join(dataroot, "hulls_val_input.npy"), val_meshes)
    np.save(os.path.join(dataroot, "hulls_val_target.npy"), val_volumes)
    np.save(os.path.join(dataroot, "hulls_test_input.npy"), test_meshes)
    np.save(os.path.join(dataroot, "hulls_test_target.npy"), test_volumes)
