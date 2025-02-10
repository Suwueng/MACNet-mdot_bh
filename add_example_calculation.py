import os
import json
import torch

import numpy as np
import pandas as pd
import rawdata_process as rdp
import network as net
import preprocess as pre

from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import r2_score, explained_variance_score
from typing import Callable, Optional, Tuple, List

load_dotenv()
DATABASE_PATH = os.getenv("DATABASE_PATH")
ROOT_PATH = os.getenv("ROOT_PATH")

if not DATABASE_PATH or not ROOT_PATH:
    raise EnvironmentError("Environment variables 'DATABASE_PATH' and 'ROOT_PATH' must be set.")


def macnet_mbh_predict(model_path: str, dataset: rdp.GalaxyDataSet, mbh: float,
                       preprocessor: pre.Preprocess) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single experiment of training and evaluating a given network model.

    Parameters
    ----------
    model_path : str
        The path to the trained model.
    dataset : rdp.GalaxyDataSet
        The dataset instance.
    preprocessor : pre.Preprocess
        The preprocessor instance.

    Returns
    -------
    y_pred_inverse_scaled : np.ndarray
        The predicted values after inverse scaling.
    dataset.mdot_macer : np.ndarray
        The true mdot_macer values from the dataset.
    dataset.mdot_bondi : np.ndarray
        The mdot_bondi values from the dataset.
    """
    data = dataset.get_attr(
        ["v1", "v2", "v3", "density", "gas_energy", "dnewstar", "temperature", "volume", "mass", "r", "theta"])
    mbh = np.log10(5e8)
    data = preprocessor.channel_standard_scaler.fit_transform(data)
    data = preprocessor.splicing(data)
    data = torch.tensor(data, dtype=torch.float32)

    model = net.MACNetRes_mbh(in_channels=data.shape[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        y_pred = model(data, mbh).detach().numpy().reshape(-1)

    y_pred_inverse_scaled = preprocessor.labels_inverse_scale(y_pred)

    return y_pred_inverse_scaled, dataset.mdot_macer, dataset.mdot_bondi


if __name__ == '__main__':
    dir_path = "/home/peng/MyFile/Projects/MACNet/results/20241216_190118_dg_MACNetRes_mbh/MACNetRes_mbh"
    dataset = rdp.GalaxyDataSet().read_hdf5(os.path.join(ROOT_PATH, 'dataset', 'filtered', 'dg_10'))
    preprocessor = pre.Preprocess(labels_scale_mode='lg')

    results = {}
    for i, model_path in tqdm(enumerate(os.listdir(dir_path)), desc='Predicting', total=len(os.listdir(dir_path))):
        y_pred, y_true, y_bondi = macnet_mbh_predict(os.path.join(dir_path, model_path), dataset, 5e8, preprocessor)
        results[f'Exp.{i:03d}'] = y_pred
        
    results.update({'y_true': y_true, 'y_bondi': y_bondi})
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(os.path.dirname(dir_path), 'results_dg_10.csv'), index=False)
