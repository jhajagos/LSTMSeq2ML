import os
import random

import h5py


class HDF5Generator:
    """Yield samples from HDF5 datasets.

    Parameters
    ----------
    filepath : str, path to HDF5 file.
    datasets : list or tuple, datasets over which to iterate.
        Datasets must have same length in first dimension.
    shuffle : bool, whether to yield samples in shuffled order.

    Examples
    --------
    >>> gen = HDF5Generator("data.h5", ["x", "y"], shuffle=True)
    >>> dset = tf.data.Dataset.from_generator(
            gen, output_types=(tf.float32, tf.float32), output_shapes=([10, 10], [5]))
    >>> next(iter(dset))
    """

    def __init__(self, filepath, datasets, shuffle=True):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"file not found: {filepath}")
        with h5py.File(filepath, mode="r") as f:
            sizes = []
            for dataset in datasets:
                if dataset not in f:
                    raise ValueError(f"dataset not found: {dataset}")
                sizes.append(f[dataset].shape[0])
        if len(set(sizes)) > 1:
            raise ValueError(
                "datasets do not have the same length in first dimension:"
                f" {', '.join(map(str, sizes))}."
            )
        self.filepath = filepath
        self.datasets = datasets
        self.shuffle = shuffle
        self.size = sizes[0]

    def __call__(self):
        indices = list(range(self.size))
        if self.shuffle:
            random.shuffle(indices)
        with h5py.File(self.filepath, mode="r") as f:
            for index in indices:
                yield tuple(f[dataset][index][:] for dataset in self.datasets)
