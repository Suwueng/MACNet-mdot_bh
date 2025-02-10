import os
import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from preprocess import log_summary, log_print, log_loss


def training(train_loader=None, val_loader=None, test_loader=None, model=None, mbh=None, criterion=None, optimizer=None,
             patience: int = None, num_epochs: int = None, batch_verbose: bool = True, device: str = None,
             save_path: str = None, logger=None, mode='single') -> nn.Module:
    """
    Training function for a PyTorch model.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader or None
        DataLoader for validation data.
    test_loader : torch.utils.data.DataLoader or None
        DataLoader for testing data.
    model : nn.Module
        The model to train.
    mbh : float
        If float, use the MBH feature with the specified value.
    criterion : nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    patience : int
        Patience for early stopping.
    num_epochs : int
        Number of epochs to train.
    batch_verbose : bool
        Whether to print batch-level logs.
    device : str
        Device to use ('cpu' or 'cuda').
    save_path : str
        Directory to save model and logs.
    logger : logging.Logger or None
        Logger object.
    mode : str or int
        If 'single', single run; if int, treated as experiment ID for naming.

    Returns
    -------
    model : nn.Module
        The trained model.
    """

    # Ensure device is set
    device = torch.device(device if device else 'cpu')
    model = model.to(device)

    # Initialize early stopper if needed
    early_stopper = EarlyStopper(patience=patience, path=save_path, logger=logger, mode=mode) if patience else None

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Check if train_loader is not empty
    if train_loader is None or len(train_loader) == 0:
        raise ValueError("train_loader is empty or None, cannot start training.")

    # Log model summary if in 'single' mode
    if mode == 'single':
        first_batch = next(iter(train_loader))
        if not isinstance(first_batch, (list, tuple)) or len(first_batch) < 1:
            raise ValueError("train_loader does not provide a (data, target) tuple.")
        log_summary(logger=logger, model=model, input_data=first_batch[0], device=device)

    train_losses = []
    val_losses = []

    # Start training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data) if mbh is None else model(data, mbh)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if mode == 'single' and batch_verbose and batch_idx % 10 == 0:
                log_print(
                    logger, f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        if mode == 'single':
            log_print(logger, f'====> Epoch: {epoch} Average loss: {train_loss:.10f}')

        # Validation step
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data) if mbh is None else model(data, mbh)
                    val_loss += criterion(output, target.view(-1, 1)).item()

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if mode == 'single':
                log_print(logger, f'====> Validation set loss: {val_loss:.10f}')

            # Early stopping check
            if early_stopper is not None:
                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    if mode == 'single':
                        log_print(logger=logger, message='Early Stopping')
                    break

        scheduler.step()

    # Save model if early stopping not triggered
    if early_stopper is None:
        _save_model(model, save_path, mode, logger)

    # Save training and validation loss logs
    training_loss = {'train_losses': train_losses}
    if val_loader is not None:
        training_loss['val_losses'] = val_losses

    _save_loss_log(training_loss, save_path, mode, logger, prefix='training')

    # Testing phase
    if test_loader is None or len(test_loader) == 0:
        if mode == 'single':
            log_print(logger, "No test_loader provided or it's empty. Skipping test evaluation.")
        return model

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) if mbh is None else model(data, mbh)
            test_loss += criterion(output, target.view(-1, 1)).item()

    test_loss /= len(test_loader.dataset)
    if mode == 'single':
        log_print(logger, f'Test Loss: {test_loss}')

    testing_loss = {'test_loss': test_loss}
    _save_loss_log(testing_loss, save_path, mode, logger, prefix='testing')

    return model


def _save_model(model: nn.Module, save_path: str, mode, logger):
    """
    Save the model state dictionary to the specified path.
    If mode == 'single', use a simple filename.
    If mode is integer, use 'ExpXXX' format.
    """
    os.makedirs(save_path, exist_ok=True)
    if mode == 'single':
        filename = f'{model.__class__.__name__}.pth'
        msg = f'Model and loss log saved in:\n\t{save_path}'
    else:
        # Attempt to treat mode as integer, otherwise just use string
        if isinstance(mode, int):
            filename = f'{model.__class__.__name__}/Exp{mode:03d}.pth'
            os.makedirs(os.path.join(save_path, f'{model.__class__.__name__}'), exist_ok=True)
        else:
            filename = f'{model.__class__.__name__}/Exp_{mode}.pth'
            os.makedirs(os.path.join(save_path, f'{model.__class__.__name__}'), exist_ok=True)
        msg = f'Model saved in:\n\t{os.path.join(save_path, filename)}'

    torch.save(model.state_dict(), os.path.join(save_path, filename))
    if logger:
        log_print(logger, msg)


def _save_loss_log(loss_data: dict, save_path: str, mode, logger, prefix='training'):
    """
    Save loss log to JSON file.
    """
    os.makedirs(save_path, exist_ok=True)
    if mode == 'single':
        filepath = os.path.join(save_path, f'{prefix}_loss.json')
    else:
        # Attempt to treat mode as integer, otherwise just use string
        if isinstance(mode, int):
            loss_dir = os.path.join(save_path, 'loss')
            os.makedirs(loss_dir, exist_ok=True)
            filepath = os.path.join(loss_dir, f'Exp{mode:03d}_{prefix}.json')
        else:
            loss_dir = os.path.join(save_path, 'loss')
            os.makedirs(loss_dir, exist_ok=True)
            filepath = os.path.join(loss_dir, f'Exp_{mode}_{prefix}.json')

    log_loss(loss_data, filepath)


## ==================================== Module ====================================


class EarlyStopper:

    def __init__(self, patience=20, verbose=True, delta=0, path=None, logger=None, mode='single'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.mode = mode
        self._logger = logger

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log_print(logger=self._logger, message=f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose and self.mode == 'single':
            log_print(
                logger=self._logger,
                message=f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')

        filename = self._get_filename(model)
        torch.save(model.state_dict(), os.path.join(self.path, filename))
        self.val_loss_min = val_loss

    def _get_filename(self, model: nn.Module) -> str:
        if self.mode == 'single':
            return f'{model.__class__.__name__}.pth'
        else:
            # If mode is integer
            if isinstance(self.mode, int):
                return f'Exp{self.mode:03d}_{model.__class__.__name__}.pth'
            else:
                return f'Exp_{self.mode}_{model.__class__.__name__}.pth'


class MACNetConv(nn.Module):

    def __init__(self):
        super().__init__()
        # Define the Convolutional blocks
        block = []
        in_channels = 15
        conv_arch = ((1, 16), (1, 32))
        for (num_convs, out_channels) in conv_arch:
            block.append(conv_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*block)

        # Define the Linear blocks
        block = []
        in_features = in_channels * 4 * 2
        linear_arch = (128, 64, 32, 1)
        for i, out_features in enumerate(linear_arch):
            relu = i < len(linear_arch) - 1
            drop = 0.3 if i < len(linear_arch) - 1 else 0.0
            block.append(linear_block(in_features, out_features, relu=relu, drop=drop))
            in_features = out_features
        self.linear_blocks = nn.Sequential(*block)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.linear_blocks(x)

        return x


class MACNetVGG(nn.Module):

    def __init__(self):
        super().__init__()
        # Define the VGG blocks
        blocks = []
        in_channels = 15
        conv_arch = ((2, 16), (4, 256))
        for (num_convs, out_channels) in conv_arch:
            blocks.append(conv_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.vgg_blocks = nn.Sequential(*blocks)

        # Define the linear blocks
        blocks = []
        in_features = in_channels * 4 * 2
        linear_arch = (in_features, in_features, in_features, 1)
        for i, out_features in enumerate(linear_arch):
            relu = i < len(linear_arch) - 1  # No ReLU in the last layer
            drop = 0.3 if i < len(linear_arch) - 1 else 0.0  # No Dropout in the last layer
            blocks.append(linear_block(in_features, out_features, relu=relu, drop=drop))
            in_features = out_features
        self.linear_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.vgg_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.linear_blocks(x)
        return x


class MACNetCConv(nn.Module):

    def __init__(self):
        super().__init__()
        # Define the Channel Linear blocks
        blocks = []
        in_channels = 15
        channel_linear_arch = (16, 32, 128)
        for i, out_channels in enumerate(channel_linear_arch):
            blocks.append(channel_linear_block(in_channels, out_channels, relu=True, drop=0.0))
            in_channels = out_channels
        self.channel_linear_blocks = nn.Sequential(*blocks)

        # Define the Convolutional blocks
        blocks = []
        conv_arch = ((1, 256), (1, 512))
        for (num_convs, out_channels) in conv_arch:
            blocks.append(conv_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*blocks)

        # Define the Linear blocks
        block = []
        in_features = in_channels * 4 * 2
        linear_arch = (4096, 4096, 1)
        for i, out_features in enumerate(linear_arch):
            relu = i < len(linear_arch) - 1
            drop = 0.3 if i < len(linear_arch) - 1 else 0.0
            block.append(linear_block(in_features, out_features, relu=relu, drop=drop))
            in_features = out_features
        self.linear_blocks = nn.Sequential(*block)

    def forward(self, x):
        x = self.channel_linear_blocks(x)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.linear_blocks(x)

        return x


class MACNetCVGG(nn.Module):

    def __init__(self):
        super().__init__()
        in_channels = 15

        # Define the Convolutional blocks
        blocks = []
        conv_arch = ((1, 256), (1, 512))
        for (num_convs, out_channels) in conv_arch:
            blocks.append(conv_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.vgg_blocks = nn.Sequential(*blocks)

        # Define the Linear blocks
        block = []
        in_features = in_channels * 4 * 2
        linear_arch = (4096, 4096, 1)
        for i, out_features in enumerate(linear_arch):
            relu = i < len(linear_arch) - 1
            drop = 0.3 if i < len(linear_arch) - 1 else 0.0
            block.append(linear_block(in_features, out_features, relu=relu, drop=drop))
            in_features = out_features
        self.linear_blocks = nn.Sequential(*block)

    def forward(self, x):
        # x = self.cvgg_blocks(x)
        x = self.vgg_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.linear_blocks(x)
        return x


class MACNetRes(nn.Module):

    def __init__(self, in_channels=15):
        super().__init__()
        in_channels = in_channels
        # Define the Residual blocks
        blocks = []
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False))
        for (num_residuals, out_channels, half) in res_arch:
            # print(in_channels, out_channels, num_residuals, first_block)
            blocks.append(
                resnet_block(num_residuals=num_residuals, in_channels=in_channels, out_channels=out_channels,
                             half=half))
            in_channels = out_channels
        self.res_blocks = nn.Sequential(*blocks)

        # Define the Linear blocks
        self.liner_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                         nn.Linear(in_channels, in_channels // 2), nn.Linear(in_channels // 2, 1))

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.liner_block(x)
        return x


class MACNetRes_mbh(nn.Module):

    def __init__(self, in_channels=15):
        super().__init__()
        in_channels = in_channels
        # Define the Residual blocks
        blocks = []
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False))
        for (num_residuals, out_channels, half) in res_arch:
            # print(in_channels, out_channels, num_residuals, first_block)
            blocks.append(
                resnet_block(num_residuals=num_residuals, in_channels=in_channels, out_channels=out_channels,
                             half=half))
            in_channels = out_channels
        self.res_blocks = nn.Sequential(*blocks)

        # Define the Linear blocks
        self.flatten_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.liner_block = nn.Sequential(nn.Linear(in_channels + 1, in_channels // 2), nn.Linear(in_channels // 2, 1))

    def forward(self, x, mbh):
        mbh = torch.tensor(mbh, dtype=torch.float32, device=x.device)
        mbh = mbh.unsqueeze(0).expand(x.size(0), -1)

        x = self.res_blocks(x)
        x = self.flatten_block(x)
        x = torch.cat((x, mbh), dim=1)
        x = self.liner_block(x)
        return x


## ================================= Module-block =================================


def conv_block(num_convs=None, in_channels=None, out_channels=None, kernel_size=3, pooling=True):
    layers = []
    if kernel_size != 1:
        padding = int((kernel_size - 1) / 2)
    else:
        padding = 0
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.LeakyReLU())
        in_channels = out_channels
    if pooling:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def linear_block(in_features, out_features, relu=True, drop=0.0):
    layers = [nn.Linear(in_features, out_features)]
    if relu:
        layers.append(nn.LeakyReLU())
    if drop > 0:
        layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)


class ChannelLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        batch_size, in_features, height, width = x.size()
        assert in_features == self.in_features, f'Input features {in_features} != {self.in_features}'

        # Flatten the height and width dimensions
        x = x.contiguous().view(batch_size, in_features, -1).permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(batch_size, self.out_features, height, width)

        return x


def channel_linear_block(in_features, out_features, relu=True, drop=0.0):
    layers = [ChannelLinear(in_features, out_features)]
    if relu:
        layers.append(nn.LeakyReLU())
    if drop > 0:
        layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)


class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(Y + X)


def resnet_block(num_residuals, in_channels, out_channels, half=False):
    layers = []
    for i in range(num_residuals):
        if i == 0:
            if half:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True))
        else:
            layers.append(Residual(out_channels, out_channels))
    return nn.Sequential(*layers)


# ==================================== Loss Function ====================================


class MSLELoss(nn.Module):

    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # log transform with epsilon to avoid log(0)
        log_pred = torch.log10(pred + self.epsilon)
        log_target = torch.log10(target + self.epsilon)
        return torch.mean((log_pred - log_target)**2)


class RELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        error = torch.sqrt((pred - target)**2)
        return torch.mean(error)
