import os
import json
import sys
import logging
import warnings
import torch

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary


class Preprocess:
    """
    Preprocess class for loading and preprocessing data
    """

    def __init__(self, device: str = 'cpu', batch_size: int = 64, valid_size: float = 0.2, test_size: float = 0.2,
                 labels_scale_mode: Optional[str] = None):
        """
        Parameters
        ----------
        device : str
            The device to move tensors onto ('cpu' or 'cuda').
        batch_size : int
            Batch size for the DataLoader.
        valid_size : float
            Fraction of data to be used as validation set.
        test_size : float
            Fraction of data to be used as test set.
        labels_scale_mode : Optional[str]
            The mode of label scaling. If None, no scaling. If 'lg', use log10 scaling.
        """
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.device = device
        self.labels_scale_mode = labels_scale_mode
        self.channel_standard_scaler = StandardScalerChannel()

        # Using a dictionary-based dispatch for label scaling reduces repetitive code
        self.scale_funcs = {None: lambda x: x, 'lg': lambda x: np.log10(x)}
        self.inverse_scale_funcs = {None: lambda x: x, 'lg': lambda x: 10**x}

    def split_data(
            self, data: np.ndarray, labels: np.ndarray,
            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and testing sets.
        """
        if not 0 < self.valid_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("valid_size and test_size must be between 0 and 1.")

        combined_test_size = self.valid_size + self.test_size
        if combined_test_size >= 1:
            raise ValueError("The sum of valid_size and test_size must be less than 1.")

        data_train, data_temp, labels_train, labels_temp = train_test_split(data, labels, test_size=combined_test_size,
                                                                            random_state=random_state)

        data_valid, data_test, labels_valid, labels_test = train_test_split(
            data_temp, labels_temp, test_size=self.test_size / combined_test_size, random_state=random_state)

        return data_train, data_valid, data_test, labels_train, labels_valid, labels_test

    def tensor(self, data: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert data and labels to tensors and move them to the specified device.
        """
        return (torch.tensor(data, dtype=torch.float32,
                             device=self.device), torch.tensor(labels, dtype=torch.float32, device=self.device))

    def create_data_loader(self, data: np.ndarray, labels: np.ndarray) -> DataLoader:
        """
        Create a PyTorch DataLoader for the given data and labels.
        If the scaler is not yet fitted, fit it first.
        """
        if self.channel_standard_scaler._mean is None or self.channel_standard_scaler._std is None:
            self.channel_standard_scaler.fit(data)

        data_tensor, labels_tensor = self.tensor(data, labels)
        dataset = CustomDataset(data_tensor, labels_tensor, device=self.device)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def labels_scale(self, labels: np.ndarray) -> np.ndarray:
        """
        Scale the labels according to the configured mode.
        """
        if self.labels_scale_mode not in self.scale_funcs:
            raise ValueError(f'Invalid mode: {self.labels_scale_mode}')
        return self.scale_funcs[self.labels_scale_mode](labels)

    def labels_inverse_scale(self, labels: np.ndarray) -> np.ndarray:
        """
        Inverse scale the labels according to the configured mode.
        """
        if self.labels_scale_mode not in self.inverse_scale_funcs:
            raise ValueError(f'Invalid mode: {self.labels_scale_mode}')
        return self.inverse_scale_funcs[self.labels_scale_mode](labels)

    @staticmethod
    def splicing(data: np.ndarray) -> np.ndarray:
        """
        Splice the data along one dimension by flipping and concatenating.
        """
        if data.ndim == 3:
            data_flipped = data[:, ::-1, :]
            return np.concatenate((data_flipped, data), axis=1)
        elif data.ndim == 4:
            data_flipped = data[:, :, ::-1, :]
            return np.concatenate((data_flipped, data), axis=2)
        else:
            raise ValueError("The input data must be a 3D or 4D array.")


class StandardScalerChannel:
    """
    Standardize each channel's features by removing the mean and scaling to unit variance.
    Expect data shape: (N, C, H, W) where C > 1.
    """

    def __init__(self):
        self._mean = None
        self._std = None
        self._epsilon = 1e-9

    def fit(self, x: np.ndarray, epsilon: Optional[float] = None) -> 'StandardScalerChannel':
        """
        Compute the mean and standard deviation of each channel.
        """
        self._validate_input_dim(x)
        self._epsilon = epsilon if epsilon is not None else self._epsilon
        self._reset()

        self._mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        self._std = np.std(x, axis=(0, 2, 3), keepdims=True)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize each channel's features.
        """
        self._validate_fit(x)
        return (x - self._mean) / (self._std + self._epsilon)

    def fit_transform(self, x: np.ndarray, epsilon=None) -> np.ndarray:
        self.fit(x, epsilon)
        return self.transform(x)

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get the mean, standard deviation, and epsilon.
        """
        return self._mean.ravel(), self._std.ravel(), self._epsilon

    def _reset(self):
        self._mean = None
        self._std = None

    @staticmethod
    def _validate_input_dim(data: np.ndarray):
        if data.ndim != 4:
            raise ValueError("The input data must be a 4-dimensional array (N, C, H, W).")

    def _validate_fit(self, data):
        """
        Validate the input data.
        """
        self._validate_input_dim(data)
        if self._mean is None or self._std is None:
            raise ValueError("StandardScalerChannel must be fitted before calling transform.")
        if data.shape[1] != self._mean.shape[1]:
            raise ValueError("Number of channels in input does not match the fitted scaler.")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # Add a batch dimension if missing
        if data.ndim == 3:
            data_transformed = data[np.newaxis, ...]
        data_transformed = self.transform(data_transformed)
        return data_transformed.reshape(data.shape)


class CustomDataset(Dataset):
    """
    Custom dataset class that applies transformations to each sample.
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform: Optional[List[Callable]] = None,
                 device='cpu'):
        """
        Parameters
        ----------
        data : torch.Tensor
            The input data tensor of shape (N, C, H, W).
        labels : torch.Tensor
            The labels tensor of shape (N, ...).
        transform : Optional[List[Callable]]
            A list of transformation functions to apply to each sample's data.
        """
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The number of samples in data and labels must be the same.")

        self.data = data
        self.labels = labels
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_data = self.data[idx]
        sample_label = self.labels[idx]

        if self.transform:
            # Apply each transform sequentially
            for t in self.transform:
                # Some transforms may return np.ndarray, convert back to tensor if needed
                sample_data = t(sample_data)

        if isinstance(sample_data, np.ndarray):
            sample_data = torch.tensor(sample_data, dtype=torch.float32, device=self.device)

        return sample_data, sample_label


class Logger:
    """
    Logger class for logging
    """

    def __init__(self, name: Optional[str] = None, save_path: str = 'app.log', level: int = logging.DEBUG):
        """
        Initialize the Logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid adding multiple handlers if logger is reused
        if not self.logger.handlers:
            file_handler = logging.FileHandler(save_path)
            file_handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


class LoggerTraining(Logger):
    """
    Specialized logger for training process which logs to file and console.
    """

    def __init__(self, name: str = 'TrainingLogger', log_file: str = 'training.log', level: int = logging.INFO):
        super().__init__(name, log_file, level)

        # Create console handler if not exists
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)


def log_print(logger: Optional[logging.Logger] = None, message: Optional[str] = None):
    """
    Print or log a message.
    """
    if message is not None:
        if logger is not None:
            logger.info(message)
        else:
            print(message)


def log_summary(logger: Optional[logging.Logger] = None, model: Optional[torch.nn.Module] = None,
                input_size: Optional[Tuple] = None, input_data: Optional[torch.Tensor] = None,
                device: Optional[str] = None, verbose: int = 0):
    """
    Print model summary using torchinfo, and log it if logger is provided.
    """
    if model is None:
        raise ValueError("A model must be provided for log_summary.")

    summary_str = summary(model, input_size=input_size, input_data=input_data, device=device, verbose=verbose)
    summary_message = f'The summary of {model.__class__.__name__} network\n{summary_str}'
    log_print(logger, summary_message)


def log_loss(log_data: Dict, save_path: str):
    """
    Save log data as JSON.
    """
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=4)


class TableFormat:
    """
    Table format class for displaying data in a formatted table.
    """

    def __init__(self, width: int = 90, line: str = '=', alignment: str = '<'):
        self._table = ''
        self._header = []
        self._header_str = []
        self._data = []
        self._data_str = []
        self._column_width = []

        self._width = width
        self._alignment = alignment
        self._line_char = line
        self._line = self._make_line()

    def get_table(self) -> str:
        return self._table

    def _make_line(self) -> str:
        return (self._line_char * (self._width // len(self._line_char)) +
                self._line_char[:(self._width % len(self._line_char))] + '\n')

    def _adjust_column_width(self):
        grid_widths = [[len(col) for col in self._header_str]]
        for row in self._data_str:
            if len(row) != len(self._header_str):
                raise ValueError("The number of columns in the header and the data does not match.")
            grid_widths.append([len(col) for col in row])

        # column-wise max width
        self._column_width = [max(col_widths) for col_widths in zip(*grid_widths)]

    def _adjust_width(self, width: Optional[int]):
        if width is not None:
            self._width = width
            self._line = self._make_line()

        self._adjust_column_width()

        # Calculate total minimum width needed
        total_min_width = sum(self._column_width) + 2 * (len(self._header_str) - 1)
        if self._width < total_min_width:
            warnings.warn(f'The current width {self._width} is insufficient, needs at least {total_min_width}. '
                          'Adjusting width to fit.')
            self._width = total_min_width
            self._line = self._make_line()

        # Distribute extra space if available
        extra_space = self._width - total_min_width
        for i in range(extra_space):
            self._column_width[i % len(self._column_width)] += 1

    def generate(self, header: List[str], data: List[List[str]], width: Optional[int] = None,
                 line: Optional[str] = None, alignment: str = '<') -> "TableFormat":
        """
        Generate a formatted table with the given header and data.
        """
        if not header:
            raise ValueError("Header cannot be empty.")

        self._header = header
        self._header_str = [str(col) for col in header]
        self._data = data
        self._data_str = [[str(col) for col in row] for row in data]

        # Update formatting parameters
        self._alignment = alignment
        if line is not None:
            self._line_char = line
            self._line = self._make_line()

        self._adjust_width(width)

        # Construct table
        # Header
        self._table = self._line
        self._table += ''.join(f'{col:{self._alignment}{w}}' for col, w in zip(self._header_str, self._column_width))
        self._table += '\n' + self._line

        # Data rows
        for row in self._data_str:
            self._table += ''.join(f'{col:{self._alignment}{w}}' for col, w in zip(row, self._column_width)) + '\n'
        self._table += self._line

        return self

    def add_note(self, note: str) -> "TableFormat":
        """
        Add a note under the table.
        """
        self._table += note + '\n' + self._line
        return self
