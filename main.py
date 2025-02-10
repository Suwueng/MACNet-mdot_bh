import os
import json
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

import rawdata_process as rdp
import network as net
import preprocess as pre

from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import r2_score, explained_variance_score
from typing import Callable, Optional, Tuple, List
"""
The mass of black hole:
Disk galaxy simulation          5e7 M_sun
Elliptical galaxy simulation    4.5e9 M_sun
"""

load_dotenv()
DATABASE_PATH = os.getenv("DATABASE_PATH")
ROOT_PATH = os.getenv("ROOT_PATH")

if not DATABASE_PATH or not ROOT_PATH:
    raise EnvironmentError("Environment variables 'DATABASE_PATH' and 'ROOT_PATH' must be set.")


def run_experiment(network: Callable[[], nn.Module], criteria: Callable[[], nn.Module], optimizer_tmp: Callable,
                   params_set: dict, num: int, logger=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single experiment of training and evaluating a given network model.

    Parameters
    ----------
    network : callable
        A function that returns an instance of the neural network model.
    criteria : callable
        A function that returns an instance of the loss function.
    optimizer_tmp : callable
        A function (class constructor) that returns an optimizer instance.
    params_set : dict
        A dictionary containing necessary parameters for training and data loading.
    num : int
        Experiment number.
    logger : optional
        Logger instance. If None, print to stdout.

    Returns
    -------
    y_pred_inverse_scaled : np.ndarray
        The predicted values after inverse scaling.
    dataset.mdot_macer : np.ndarray
        The true mdot_macer values from the dataset.
    dataset.mdot_bondi : np.ndarray
        The mdot_bondi values from the dataset.
    """
    device = params_set['device']
    batch_size = params_set['batch_size']
    test_size = 0.2  # Hard-coded as in original code
    label_scale_mode = params_set['label_scale']

    # Data load and preprocess
    data_path = params_set['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    preprocessor = pre.Preprocess(device=device, batch_size=batch_size, test_size=test_size,
                                  labels_scale_mode=label_scale_mode)

    dataset = rdp.GalaxyDataSet().read_hdf5(data_path)
    data = dataset.get_attr() if params_set['attr_list'] is None else dataset.get_attr(params_set['attr_list'])
    labels = preprocessor.labels_scale(dataset.mdot_macer)
    mbh = np.log10(params_set['mbh'])

    data = preprocessor.channel_standard_scaler.fit_transform(data)
    data = preprocessor.splicing(data)

    # Split data
    data_train, data_valid, data_test, labels_train, labels_valid, labels_test = preprocessor.split_data(data, labels)

    # Create data loaders
    train_loader = preprocessor.create_data_loader(data_train, labels_train)
    valid_loader = preprocessor.create_data_loader(data_valid, labels_valid)
    test_loader = preprocessor.create_data_loader(data_test, labels_test)

    # Log dataset information
    dataset_message = (f"The size of datasets\n"
                       f"\ttrain set:  {len(data_train):>6}\n"
                       f"\tvalid set:  {len(data_valid):>6}\n"
                       f"\ttest set:   {len(data_test):>6}")
    _log_info(logger, dataset_message, condition=(num == 0))

    # Create model, criterion, and optimizer
    model = network(in_channels=data.shape[1]).to(device)
    criterion = criteria()
    optimizer = optimizer_tmp(model.parameters(), lr=params_set['learning_rate'])

    # Training
    mode = 'single' if num == 0 else num
    model = net.training(train_loader=train_loader, val_loader=valid_loader, test_loader=test_loader, model=model,
                         mbh=mbh, criterion=criterion, optimizer=optimizer, patience=params_set['patience'],
                         num_epochs=params_set['num_epochs'], batch_verbose=params_set['batch_verbose'], device=device,
                         save_path=params_set['save_path'], logger=logger, mode=num)

    # Evaluation on full data
    model.eval()
    data_tensor, _ = preprocessor.tensor(data, labels)
    with torch.no_grad():
        if mbh is None:
            y_pred = model(data_tensor).detach().cpu().numpy().reshape(-1)
        else:
            y_pred = model(data_tensor, mbh).detach().cpu().numpy().reshape(-1)
    y_pred_inverse_scaled = preprocessor.labels_inverse_scale(y_pred)

    return y_pred_inverse_scaled, dataset.mdot_macer, dataset.mdot_bondi


def _log_info(logger, message: str, condition: bool = True):
    """
    Log or print a message if condition is True.

    Parameters
    ----------
    logger : optional
        Logger instance. If None, print to stdout.
    message : str
        The message to log or print.
    condition : bool
        If True, log or print the message; otherwise do nothing.
    """
    if not condition:
        return
    if logger is not None:
        logger.info(message)
    else:
        print(message)


if __name__ == "__main__":
    # Configuration
    num_experiments = 200
    galaxy_type = 'dg'
    Network = net.MACNetRes_mbh
    Criteria = nn.MSELoss
    Optimizer = optim.Adam

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("results", f"{timestamp}_{galaxy_type}_{Network.__name__}")

    # Ensure save directory exists
    os.makedirs(os.path.join(ROOT_PATH, save_dir), exist_ok=True)

    params = {
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        'model': Network.__name__,
        'criteria': Criteria.__name__,
        'optimizer': Optimizer.__name__,
        'save_path': os.path.join(ROOT_PATH, save_dir),
        'random_seed': 28,
        'batch_size': 64,
        'num_epochs': 1000,
        'learning_rate': 0.001,
        'patience': None,
        'batch_verbose': False,
        'label_scale': 'lg',
        'data_path': os.path.join(ROOT_PATH, 'dataset', 'filtered', galaxy_type),
        'attr_list':
        ["v1", "v2", "v3", "density", "gas_energy", "dnewstar", "temperature", "volume", "mass", "r", "theta"]
    }

    if galaxy_type == 'eg':
        params['mbh'] = 4.5e9
    elif galaxy_type == 'dg':
        params['mbh'] = 5e7
    elif galaxy_type == 'dg_10':
        params['mbh'] = 5e8

    # Save parameters to file
    logger_train = pre.LoggerTraining(log_file=os.path.join(params['save_path'], 'training.log')).get_logger()
    pre.log_print(logger_train, message=f'The experiment configure:\n{json.dumps(params, indent=4)}')

    #  # Run single experiment (num=0 as example)
    # y_pred, y_true, y_bondi = run_experiment(Network, Criteria, Optimizer, params, num=0, logger=None)

    results = {}
    results_stat = []
    for i in range(num_experiments):
        pre.log_print(logger_train, f"Running experiment {i+1}/{num_experiments}")
        y_pred, y_true, y_bondi = run_experiment(Network, Criteria, Optimizer, params, num=i, logger=logger_train)
        # print(min(y_bondi), max(y_bondi))
        results.update({f'Exp.{i+1:03d}': y_pred})
        results_stat.append([
            np.mean((np.log10(y_pred) - np.log10(y_true))**2),
            r2_score(np.log10(y_true), np.log10(y_pred)),
            explained_variance_score(np.log10(y_true), np.log10(y_pred)),
        ])
        pre.log_print(logger_train,
                      f"\tMSLE: {results_stat[-1][0]}, R2: {results_stat[-1][1]}, EVS: {results_stat[-1][2]}")

    results.update({'y_true': y_true, 'y_bondi': y_bondi})
    # print(min(y_bondi), max(y_bondi))
    results = pd.DataFrame(results)
    results_save_path = os.path.join(params['save_path'], 'results.csv')
    results.to_csv(results_save_path, index=False)

    results_stat = np.array(results_stat)
    pre.log_print(
        logger_train, message=f'The results of experiments are saved in {results_save_path}\n'
        f'The statistics of results are:\n'
        f'MSLE: {np.mean(results_stat[:, 0])} ± {np.std(results_stat[:, 0])}\n'
        f'R2: {np.mean(results_stat[:, 1])} ± {np.std(results_stat[:, 1])}\n'
        f'EVS: {np.mean(results_stat[:, 2])} ± {np.std(results_stat[:, 2])}\n')
