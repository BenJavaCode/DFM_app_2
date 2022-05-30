"""
Spectral dataset. For creating a dataset and wrapping it in a dataloader.
"""


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np


class SpectralDataset(Dataset):
    """
    Description: Builds a dataset of spectral data. Use idxs to specify which samples to use
                 for dataset - this allows for random splitting into training, validation,
                 and test sets. Instead of passing in filenames for X and y, we can also
                 pass in numpy arrays directly.
    Latest update: 14-06-2021. Added more comments.
    """
    def __init__(self, X_fn, y_fn, idxs=None, transform=None):
        """
        __init__(X_fn, y_fn, idxs, transform)
        Description: Initializes dataset params.
        Params: X_fn = input data.
                y_fn = input data labels.
                idxs = List of indices, that will be used for this dataset.
                transforms = Composition of transformations, that will be applyed to data(X_fn)
        Latest update: 14-06-2021. Added more comments.
        """

        # LOAD FROM STRING OR DIRECTLY FROM ARRAY
        if type(X_fn) == str:
            self.X = np.load(X_fn)
        else:
            self.X = X_fn
        if type(y_fn) == str:
            self.y = np.load(y_fn)
        else:
            self.y = y_fn
        # -

        if idxs is None: idxs = np.arange(len(self.y))
        self.idxs = idxs
        self.transform = transform  # For possible custom transforms.

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):  # Get data-point by index.
        i = self.idxs[idx]
        x, y = self.X[i], self.y[i]
        x = np.expand_dims(x, axis=0)  # Add new dimension at dim 0. eg. [1,2].expand(axis=0) = [[1,2]]
        # Apply transformation to datapoint.
        if self.transform:
            x = self.transform(x)
        return (x, y)


# Transformation functions
# -----------------------------------

class GetInterval(object):
    """
    GetInterval(object)
    Description: Gets an interval of each spectrum. Used if subspace of datapoints is needed.
    Latest update: 14-06-2021. Added more comments.
    """
    def __init__(self, min_idx, max_idx):
        self.min_idx = min_idx
        self.max_idx = max_idx

    def __call__(self, x):
        # This means for each subspace, in x, [min:max]
        # Eg. for array x = [[1,2,3], [1,2,3]]. x[:,0:2] = [[1,2], [1,2]]
        x = x[:,self.min_idx:self.max_idx]
        return x


class ToFloatTensor(object):
    """
    ToFloatTensor(object)
    Description: Converts numpy arrays to float Variables in Pytorch.
    Latest update: 14-06-2021. Added more comments.
    """
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        return x


class AddStripAug(object):
    """
    Explanation will come if this works
    """

    def __init__(self, back_avg, back_sampler):
        self.back_avg = back_avg
        self.back_sampler = back_sampler

    def __call__(self, x):
        np.random.shuffle(self.back_sampler)
        np.random.shuffle(self.back_avg)
        r = np.random.uniform(0, 1.4)
        x = x + (self.back_sampler[0] * r)
        back = float(x[0][479] / self.back_avg[0][479]) * self.back_avg[0]
        x = np.array(np.array(x) - back)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        return x


def spectral_dataloader(X_fn, y_fn, idxs=None, batch_size=10, shuffle=True,
                        num_workers=2, min_idx=None, max_idx=None, sampler=None, back_avg=None, back_sampler=None):
    """
    spectral_dataloader(X_fn, y_fn, idxs=None, batch_size, shuffle,
    num_workers, min_idx, max_idx, sampler)

    Description: Adds tranforms and composes them.
                 Creates dataset.
                 Returns a DataLoader with spectralDataset.
    Params: X_fn = input data.
            y_fn = input data labels.
            idxs = list containing which indices - from the dataset, that will be used.
            batch_size = number of datapoints, that will be fed to the model at a time.
            shuffle = Shuffles the data if set to True.
            num_workers = Number of sub-processes needed for loading the data.(If CPU has 2 threads, this should be 2).
            min_idx = Start of subspace from each datapoint.
            max_idx = End of subspace from each datapoint.
            sampler = A sampler defines the strategy to retrieve the sample
                      – sequential or random or any other manner. Shuffle should be set to false when a sampler is used.
    Latest update: 14-06-2021. Added more comments.
    """

    # TRANSFORMS
    transform_list = []
    if min_idx is not None and max_idx is not None:  # If you want a subspace from each datapoint.
        transform_list.append(GetInterval(min_idx, max_idx))
    if back_avg is not None and back_sampler is not None:
        transform_list.append(AddStripAug(back_avg, back_sampler))
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    # -

    dataset = SpectralDataset(X_fn, y_fn, idxs=idxs, transform=transform)

    # Create dataloader from dataset.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, sampler=sampler)
    return dataloader


