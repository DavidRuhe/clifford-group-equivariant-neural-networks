import glob
import os
from math import sqrt

import h5py
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class JetDataset(Dataset):
    """
    PyTorch dataset.
    """

    def __init__(self, data, num_pts=-1, shuffle=True):
        self.data = data

        if num_pts > len(data["Nobj"]):
            raise ValueError(
                "Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!".format(
                    num_pts, len(data["Nobj"])
                )
            )

        if num_pts < 0:
            self.num_pts = len(data["Nobj"])
        else:
            self.num_pts = num_pts

        if shuffle:
            g = torch.Generator().manual_seed(42)
            self.perm = torch.randperm(len(data["Nobj"]), generator=g)[: self.num_pts]
        else:
            self.perm = None

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}


def initialize_datasets(datadir="./data", num_pts=None):
    """
    Initialize datasets.
    """

    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the HDF5 files (e.g. out_test.h5, out_train.h5, out_valid.h5) reside.
    # There may be many data files, in some cases the test/train/validate sets may themselves be split across files.
    # We will look for the keywords defined in splits to be be in the filenames, and will thus determine what
    # set each file belongs to.
    splits = [
        "train",
        "test",
        "valid",
    ]  # Our data categories -- training set, testing set and validation set
    patterns = {
        "train": "train",
        "test": "test",
        "valid": "val",
    }  # Patterns to look for in data files, to identify which data category each belongs in

    files = glob.glob(datadir + "/*.h5")
    assert len(files) > 0, f"Could not find any HDF5 files in the directory {datadir}!"
    datafiles = {split: [] for split in splits}
    for file in files:
        for split, pattern in patterns.items():
            if pattern in file:
                datafiles[split].append(file)
    nfiles = {split: len(datafiles[split]) for split in splits}

    ### ------ 2: Set the number of data points ------ ###
    # There will be a JetDataset for each file, so we divide number of data points by number of files,
    # to get data points per file. (Integer division -> must be careful!)
    # TODO: nfiles > npoints might cause issues down the line, but it's an absurd use case
    if num_pts is None:
        num_pts = {"train": -1, "test": -1, "valid": -1}

    num_pts_per_file = {}
    for split in splits:
        num_pts_per_file[split] = []

        if num_pts[split] == -1:
            for n in range(nfiles[split]):
                num_pts_per_file[split].append(-1)
        else:
            for n in range(nfiles[split]):
                num_pts_per_file[split].append(
                    int(np.ceil(num_pts[split] / nfiles[split]))
                )
            num_pts_per_file[split][-1] = int(
                np.maximum(
                    num_pts[split] - np.sum(np.array(num_pts_per_file[split])[0:-1]), 0
                )
            )

    ### ------ 3: Load the data ------ ###
    datasets = {}
    for split in splits:
        print(f"Loading {split} data...")
        datasets[split] = []
        for file in datafiles[split]:
            with h5py.File(file, "r") as f:
                datasets[split].append(
                    {
                        key: torch.from_numpy(val[: num_pts[split]])
                        for key, val in f.items()
                    }
                )

    ### ------ 4: Error checking ------ ###
    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = []
    for split in splits:
        for dataset in datasets[split]:
            keys.append(dataset.keys())
    assert all([key == keys[0] for key in keys]), "Datasets must have same set of keys!"

    ### ------ 5: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    torch_datasets = {
        split: ConcatDataset(
            [
                JetDataset(data, num_pts=num_pts_per_file[split][idx])
                for idx, data in enumerate(datasets[split])
            ]
        )
        for split in splits
    }

    return torch_datasets


def batch_stack_general(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Unlike :batch_stack:, this will automatically stack scalars, vectors,
    and matrices. It will also automatically convert Numpy Arrays to
    Torch Tensors.

    Parameters
    ----------
    props : list or tuple of Pytorch Tensors, Numpy ndarrays, ints or floats.
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if type(props[0]) in [int, float]:
        # If batch is list of floats or ints, just create a new Torch Tensor.
        return torch.tensor(props)

    if type(props[0]) is np.ndarray:
        # Convert numpy arrays to tensors
        props = [torch.from_numpy(prop) for prop in props]

    shapes = [prop.shape for prop in props]

    if all(shapes[0] == shape for shape in shapes):
        # If all shapes are the same, stack along dim=0
        return torch.stack(props)

    elif all(shapes[0][1:] == shape[1:] for shape in shapes):
        # If shapes differ only along first axis, use the RNN pad_sequence to pad/stack.
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

    elif all((shapes[0][2:] == shape[2:]) for shape in shapes):
        # If shapes differ along the first two axes, (shuch as a matrix),
        # pad/stack first two axes

        # Ensure that input features are matrices
        assert all(
            (shape[0] == shape[1]) for shape in shapes
        ), "For batch stacking matrices, first two indices must match for every data point"

        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(
            max_shape, dtype=props[0].dtype, device=props[0].device
        )

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor
    else:
        ValueError(
            "Input tensors must have the same shape on all but at most the first two axes!"
        )


def batch_stack(props, edge_mat=False, nobj=None):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """

    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        props = [p[:nobj, ...] for p in props]
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(
            max_shape, dtype=props[0].dtype, device=props[0].device
        )

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0 or props[0].shape[0] != to_keep.shape[0]:
        return props
    else:
        return props[:, to_keep, ...]


def normsq4(p):
    # Quick hack to calculate the norms of the four-vectors
    # The last dimension of the input gets eaten up
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


enc = OneHotEncoder().fit([[-1], [1]])


def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx * n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges


def collate_fn(
    data, scale=1.0, nobj=None, edge_features=[], add_beams=False, beam_mass=1
):
    """
    Collation function that collates datapoints into the batch format for lgn

    Parameters
    ----------
    data : list of datapoints
        The data to be collated.
    edge_features : list of strings
        Keys of properties that correspond to edge features, and therefore are
        matrices of shapes (num_atoms, num_atoms), which when forming a batch
        need to be padded along the first two axes instead of just the first one.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    data = {
        prop: batch_stack([mol[prop] for mol in data], nobj=nobj)
        for prop in data[0].keys()
    }
    data["label"] = data["label"].to(torch.bool)

    # to_keep = batch['Nobj'].to(torch.uint8)
    to_keep = torch.any(data["label"], dim=0)
    data = {key: drop_zeros(prop, to_keep) for key, prop in data.items()}

    if add_beams:
        beams = torch.tensor(
            [
                [
                    [sqrt(1 + beam_mass**2), 0, 0, 1],
                    [sqrt(1 + beam_mass**2), 0, 0, -1],
                ]
            ],
            dtype=data["Pmu"].dtype,
        ).expand(data["Pmu"].shape[0], 2, 4)
        s = data["Pmu"].shape
        data["Pmu"] = torch.cat([beams * scale, data["Pmu"] * scale], dim=1)
        labels = torch.cat((torch.ones(s[0], 2), -torch.ones(s[0], s[1])), dim=1)
        if "scalars" not in data.keys():
            data["scalars"] = labels.to(dtype=data["Pmu"].dtype).unsqueeze(-1)
        else:
            data["scalars"] = torch.stack(
                (data["scalars"], labels.to(dtype=data["Pmu"].dtype))
            )
    else:
        data["Pmu"] = data["Pmu"] * scale

    # batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
    atom_mask = data["Pmu"][..., 0] != 0.0
    # Obtain edges
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    data["atom_mask"] = atom_mask.to(torch.bool)
    data["edge_mask"] = edge_mask.to(torch.bool)

    batch_size, n_nodes, _ = data["Pmu"].size()

    # Centralize Data
    # data['Pmu'] = data['Pmu'] - data['Pmu'].sum(dim=1, keepdim=True) / data['Nobj'][:,None,None]

    if add_beams:
        beamlabel = data["scalars"]
        one_hot = (
            enc.transform(beamlabel.reshape(-1, 1))
            .toarray()
            .reshape(batch_size, n_nodes, -1)
        )
        one_hot = torch.tensor(one_hot)

        mass = normsq4(data["Pmu"]).abs().sqrt().unsqueeze(-1)  # [B,N,1]
        mass_tensor = mass.view(mass.shape + (1,))
        nodes = (one_hot.unsqueeze(-1) * mass_tensor).view(
            mass.shape[:2] + (-1,)
        )  # [B,N,2]
    else:
        mass = normsq4(data["Pmu"]).unsqueeze(-1)
        nodes = mass

    edges = get_adj_matrix(n_nodes, batch_size, data["edge_mask"])
    data["nodes"] = nodes
    data["edges"] = edges

    return data


def retrieve_dataloaders(batch_size, num_workers=4, num_train=-1, datadir="./data"):
    # Initialize dataloader
    datadir = "/home/druhe/github/LorentzNet-release/data/top/"
    datasets = initialize_datasets(
        datadir, num_pts={"train": num_train, "test": -1, "valid": -1}
    )
    # distributed training
    # train_sampler = DistributedSampler(datasets['train'])
    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size
            if (split == "train")
            else batch_size,  # prevent CUDA memory exceeded
            # sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True if (split == "train") else False,
            num_workers=num_workers,
            collate_fn=collate,
        )
        for split, dataset in datasets.items()
    }

    return dataloaders


class TopTaggingDataset:
    def __init__(
        self,
        batch_size=32,
        num_train=1024,
        num_val=1024,
        num_test=1024,
        # num_workers=4,
    ) -> None:
        self.batch_size = batch_size
        # self.num_workers = num_workers
        datadir = os.path.join(os.environ["DATAROOT"], "top_tagging")
        self.datasets = initialize_datasets(
            datadir, num_pts={"train": num_train, "valid": num_val, "test": num_test}
        )
        self.collate = lambda data: collate_fn(
            data, scale=1, add_beams=True, beam_mass=1
        )

    def train_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.datasets["train"]) if distributed else None
        shuffle = False if distributed else True
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            # pin_memory=True,
            # persistent_workers=True,
            drop_last=True,
            # num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.datasets["valid"]) if distributed else None
        return DataLoader(
            self.datasets["valid"],
            batch_size=self.batch_size,
            # pin_memory=True,
            # persistent_workers=True,
            drop_last=True,
            # num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate,
        )

    def test_loader(self):
        distributed = torch.distributed.is_initialized()
        sampler = DistributedSampler(self.datasets["test"]) if distributed else None
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            # pin_memory=True,
            # persistent_workers=True,
            drop_last=True,
            # num_workers=self.num_workers,
            collate_fn=self.collate,
            sampler=sampler,
        )
