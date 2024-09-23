import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms

import prediction.network_factory as net
from prediction.analyse_factory import TrainingAnalyse, MdotResult
from prediction.logging_factory import LoggerTraining, log_print
from prediction.preprocess_factory import data_process
from toolbox import root_path

"""
The mass of black hole:
Disk galaxy simulation          5e7 M_sun
Elliptical galaxy simulation    4.5e9 M_sun
"""

# Using GPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Initial save directory file and logger
save_path = os.path.join(root_path, f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(save_path, exist_ok=True)

logger_training = LoggerTraining(log_file=os.path.join(save_path, 'training.log')).get_logger()

if __name__ == "__main__":
    # ======================== Initial parameters and model ========================
    params = {
        'random_seed': 28,
        'batch_size': 64,
        'num_epochs': 500,
        'learning_rate': 0.001,
        'patience': None,
        'batch_verbose': False,
        'shuffle': True,
        'splicing': True,
        'yscale': 'log',
        'n': 1,
        'data_path': ('dataset/coarse/dg_954/x_1.npy', 'dataset/coarse/dg_954/y_1.npy'),
        'bondi_acc_path': 'dataset/coarse/dg_954/bondi_accretion_rate_0.3.npy'
    }

    plot_key_dict = {
        'save_path': save_path,
        'logger': logger_training,
        'grid': True
    }

    # Create model
    model = net.MACNetCConv().to(device)
    loaded_model = net.MACNetCConv().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    params.update({'model': model.__class__.__name__,
                   'criterion': criterion.__class__.__name__,
                   'optimizer': optimizer.__class__.__name__})

    # Record parameter configure
    log_print(logger_training, message=f'The experiment configure:\n{json.dumps(params, indent=4)}')

    # ================================ Data process ================================
    dataset, training_set, validation_set, test_set = data_process(
        yscale=params['yscale'],
        n=params['n'],
        splicing=params['splicing'],
        path=params['data_path'],
        logger=logger_training
    )

    train_loader = training_set.get_dataloader(
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        device=device
    )
    val_loader = validation_set.get_dataloader(
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        device=device
    )
    test_loader = test_set.get_dataloader(
        batch_size=params['batch_size'],
        shuffle=params['shuffle'],
        device=device
    )

    # ================================== Training ==================================
    net.training(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        patience=params['patience'],
        num_epochs=params['num_epochs'],
        batch_verbose=params['batch_verbose'],
        device=device,
        save_path=save_path,
        logger=logger_training
    )

    # =================================== Analyse ===================================
    with open(os.path.join(save_path, 'training_loss.json')) as f:
        losses = json.load(f)

    TrainingAnalyse(**losses).loss_curve(
        title='Training Loss Curve',
        **plot_key_dict
    )

    loaded_model.load_state_dict(torch.load(
        os.path.join(save_path, f'{loaded_model.__class__.__name__}.pth'), weights_only=True
    ))

    loaded_model.eval()

    # Test set results
    y_pred = loaded_model(test_set.get_tensor(device=device)[0]).detach().cpu().numpy().reshape(-1)
    np.save(os.path.join(save_path, 'y_pred'), y_pred)

    result = MdotResult(
        target=dataset.yscale_inverse(test_set.y),
        pred=dataset.yscale_inverse(y_pred)
    )

    result.statistics(logger=logger_training)

    plot_line_key_dict = plot_key_dict.copy()
    plot_line_key_dict.update({
        'figsize': (8, 4),
        'xlabel': 'Time Frame'
    })

    result.scatter_fit(
        xscale='log',
        yscale='log',
        title='Test Set Fit Quality Scatter Plot',
        legend=False,
        **plot_key_dict
    )
    result.distribution_hist(
        figsize=(8, 4),
        yscale='log',
        title='Test Set Distribution of True and Predicted Values',
        bins=40,
        om=True,
        **plot_key_dict
    )

    # Overall dataset results
    y_pred = loaded_model(dataset.get_tensor(device=device)[0]).detach().cpu().numpy().reshape(-1)
    y_bondi = np.load(os.path.join(root_path, params['bondi_acc_path']))
    model_name = loaded_model.__class__.__name__

    result = MdotResult(
        target=dataset.yscale_inverse(),
        pred=(dataset.yscale_inverse(y_pred), *y_bondi.swapaxes(0, 1)),
        pred_label=(
            model_name,
            *[f'Bondi {i}' for i in range(len(y_bondi.swapaxes(0, 1)))]
        ),
        x = np.arange(len(y_pred))
    )
    result.statistics(
        logger=logger_training
    )
    result.plot(
        ylabel='Value',
        yscale='log',
        title=f'True vs {model_name} vs Bondi Line Plot',
        alpha=0.5,
        **plot_line_key_dict
    )
    result.plot_error_om(
        ylabel='Order of Magnitude',
        title=f'Order of Magnitude of {model_name} and Bondi Error Line Plot',
        alpha=0.5,
        # legend=False,
        **plot_line_key_dict)
    result.scatter_fit(
        xscale='log',
        yscale='log',
        title=f'{model_name} and Bondi Fit Quality Scatter Plot',
        alpha=0.3,
        # legend=False,
        **plot_key_dict
    )
    result.distribution_hist(
        figsize=(8, 4),
        yscale='log',
        title=f'Distribution of True, {model_name} and Bondi Values',
        bins=30,
        om=True,
        **plot_key_dict
    )
    result.plot_delta_m(
        figsize=(8, 4),
        yscale='log',
        title=r'$\Delta M_\text{bh}$ ' + f'of True, {model_name} and Bondi',
        legend='lower right',
        **plot_key_dict
    )
