import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from toolbox import root_path

random_seed = 728  # Setting the seed of random


class DataSet:
    def __init__(self):
        self._x = None
        self._y = None
        self._x_tensor = None
        self._y_tensor = None
        self._dataloader = None
        self._yscale = None
        self._n = None

        self.x = None
        self.y = None
        self.standard_scaler = None

    def __len__(self):
        return len(self.x)

    def load(self, data: tuple = None, path: tuple = None):
        if data is not None:
            self._x, self._y = data
        elif path is not None:
            self._x, self._y = np.load(path[0]), np.load(path[1])
        else:
            raise ValueError('The parameters data and path must have at least one value that is not None.')

        self.x, self.y = self._x, self._y

        return self

    def get_original(self):
        return self._x, self._y

    def splicing(self):
        self.x = np.transpose(np.concatenate((self._x[:, :, :, ::-1], self._x), axis=3), (0, 1, 3, 2))

        return self

    def yscale_fit_transform(self, yscale=None, n=1):
        self._yscale = yscale
        match self._yscale:
            case None:
                self.y = self.y
            case 'log':
                self._n = n
                self.y = np.log10(self._y) + self._n
            case 'exp10':
                self._n = n
                self.y *= 10 ** self._n
            case _:
                raise ValueError('The yscale must be None, log or exp10.')

        return self

    def yscale_inverse(self, y=None):
        if y is None:
            y = self.y

        match self._yscale:
            case None:
                return y
            case 'log':
                return 10 ** (y - self._n)
            case 'exp10':
                return y / 10 ** self._n
            case _:
                raise ValueError('The yscale must be None, log or exp10.')


    def standard_fit(self, epsilon=1e-8):
        self.standard_scaler = StandardScalerChannel()
        self.x = self.standard_scaler.fit(self.x, epsilon=epsilon)

        return self

    def standard_transform(self):
        self.x = self.standard_scaler.transform(self.x)

        return self

    def standard_fit_transform(self, epsilon=1e-8):
        self.standard_scaler = StandardScalerChannel()
        self.x = self.standard_scaler.fit_transform(self.x, epsilon=epsilon)

        return self

    def get_tensor(self, device='cpu'):
        self._x_tensor = torch.tensor(self.x, dtype=torch.float32).to(device)
        self._y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1).to(device)

        return self._x_tensor, self._y_tensor

    def get_dataloader(self, batch_size: int = None, shuffle: bool = True, transform=None, device='cpu'):
        if self._x_tensor is None or self._y_tensor is None:
            self.get_tensor(device)

        self._dataloader = DataLoader(
            CustomDataset(
                self._x_tensor,
                self._y_tensor,
                transform=transform
            ),
            batch_size=batch_size,
            shuffle=shuffle
        )

        return self._dataloader


def data_process(path: tuple = None, logger=None, yscale=None, n=None, splicing=None):
    # Loading data & log10(y) & Splicing into a whole
    dataset_ = DataSet().load(
        path=(os.path.join(root_path, path[0]), os.path.join(root_path, path[1]))
    )

    if splicing:
        dataset_ = dataset_.splicing()

    dataset_ = dataset_.yscale_fit_transform(yscale=yscale, n=n)
    dataset_ = dataset_.standard_fit_transform()

    # Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(
        dataset_.x, dataset_.y, test_size=0.2, random_state=random_seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=random_seed)

    # Standard
    training_set_ = DataSet().load(data=(x_train, y_train))
    validation_set_ = DataSet().load(data=(x_val, y_val))
    test_set_ = DataSet().load(data=(x_test, y_test))

    dataset_message = (f"The size of datasets\n"
                       f"\twhole dataset:  {len(dataset_):>6}\n"
                       f"\ttraining set:   {len(training_set_):>6}\n"
                       f"\tvalidation set: {len(validation_set_):>6}\n"
                       f"\ttest set:       {len(test_set_):>6}")
    if logger is not None:
        logger.info(dataset_message)
    else:
        print(dataset_message)

    return dataset_, training_set_, validation_set_, test_set_


class StandardScalerChannel:
    """
    Standardize each channel features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self._mean = None
        self._std = None
        self._epsilon = 1e-8

    def _reset(self):
        self._mean = None
        self._std = None

    def fit(self, x: np.ndarray, epsilon=None):
        if epsilon is not None:
            self._epsilon = epsilon

        self._reset()

        self._mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        self._std = np.std(x, axis=(0, 2, 3), keepdims=True)

        return self

    def transform(self, x: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("The StandardScalerChannel instance is not fitted yet. Call 'fit' with "
                                 "appropriate arguments before using this estimator.")
        elif x.shape[1] != self._mean.size:
            raise ValueError(f"The number of input channels ({x.shape[1]},) is not equal to "
                             f"the number of scaler channels ({self._mean.shape[1]},).")

        return (x - self._mean) / (self._std + self._epsilon)

    def fit_transform(self, x: np.ndarray, epsilon=None):
        self.fit(x, epsilon)

        return self.transform(x)

    def get_parameters(self):
        return self._mean.reshape(-1), self._std.reshape(-1), self._epsilon


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)

        return x, y
