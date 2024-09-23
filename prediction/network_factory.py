import os
import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from prediction.logging_factory import log_summary, log_print, log_loss


def training(train_loader=None,
             val_loader=None,
             test_loader=None,
             model=None,
             criterion=None,
             optimizer=None,
             patience: int = None,
             num_epochs=None,
             batch_verbose=True,
             device=None,
             save_path=None,
             logger=None):

    early_stopper = EarlyStopper(patience=patience, path=save_path, logger=logger) if patience else None
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    device = torch.device(device if device else  'cpu')
    log_summary(logger=logger, model=model, input_data=next(iter(train_loader))[0], device=device)

    log_print(logger, 'Starting training')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_verbose and batch_idx % 10 == 0:
                log_print(logger, f' {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        log_print(logger, f'====> Epoch: {epoch} Average loss: {train_loss:.10f}')

        if val_loader is not None:
            model.eval()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            log_print(logger, f'========> Validation loss: {val_loss:.10f}')

            if early_stopper is not None:
                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    log_print(logger=logger, message='Early Stopping')
                    break

        scheduler.step()

    if early_stopper is None:
        torch.save(model.state_dict(), os.path.join(save_path, f'{model.__class__.__name__}.pth'))
        log_print(logger, f'Model and loss log saved in:\n\t{save_path}')

    if val_loader is None:
        training_loss = {'train_losses': train_losses}
    else:
        training_loss = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    log_loss(training_loss, os.path.join(save_path, 'training_loss.json'))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader.dataset)
    log_print(logger, f'Test Loss: {test_loss}')

    testing_loss = {
        'test_loss': test_loss
    }
    log_loss(testing_loss, os.path.join(save_path, 'testing_loss.json'))

    return model


## ==================================== Module ====================================
class EarlyStopper:
    def __init__(self, patience=20, verbose=True, delta=0, path=None, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

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
        """Saves model when validation loss decrease."""
        if self.verbose:
            log_print(logger=self._logger,
                      message=f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                              f'Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, f'{model.__class__.__name__}.pth'))
        self.val_loss_min = val_loss


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
        channel_linear_arch = (in_channels, 32, 128)
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
        linear_arch = (1024, 1024, 1)
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





def conv_block(num_convs=None, in_channels=None, out_channels=None):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU())
        in_channels=out_channels
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


# ==================================== Loss Function ====================================
class MSLELoss(nn.Module):
    def __init__(self, epsilon=1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Adding 1 to avoid log(0), and to ensure the minima are still near 0
        log_pred = torch.log10(pred + self.epsilon)
        log_target = torch.log10(target + self.epsilon)
        # log_error = torch.log10((pred + self.epsilon) / (target + self.epsilon))
        return torch.mean((log_pred - log_target) ** 2)

class RELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        error = torch.sqrt((pred - target) ** 2)
        return torch.mean(error)