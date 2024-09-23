import os
import re
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error, explained_variance_score

from prediction.logging_factory import TableFormat, log_print


def save_show_wrapper():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            xlabel = kwargs.pop('xlabel', None)
            ylabel = kwargs.pop('ylabel', None)
            xscale = kwargs.pop('xscale', None)
            yscale = kwargs.pop('yscale', None)
            figsize = kwargs.pop('figsize', (5, 5))
            dpi = kwargs.pop('dpi', 300)
            title = kwargs.pop('title', None)
            legend = kwargs.pop('legend', True)
            save_path = kwargs.pop('save_path', None)
            suffix = kwargs.pop('suffix', 'png')
            show = kwargs.pop('show', True)
            logger = kwargs.pop('logger', None)
            grid =kwargs.pop('grid', False)

            plt.figure(figsize=figsize, dpi=dpi)

            func(*args, **kwargs)

            plt.grid(grid)
            if grid:
                grid_color = plt.rcParams['grid.color']

                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_color(grid_color)

                ax.tick_params(axis='both', which='both', colors=grid_color, labelcolor='black')
                if ax.get_xaxis().get_minor_locator() or ax.get_yaxis().get_minor_locator():
                    ax.tick_params(axis='both', which='minor', colors=grid_color)


            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            if xscale is not None:
                plt.xscale(xscale)
            if yscale is not None:
                plt.yscale(yscale)
            if title is not None:
                plt.title(title)
            if legend:
                if isinstance(legend, str):
                    plt.legend(loc=legend)
                else:
                    plt.legend()
            if save_path is not None:
                if title is None:
                    title = 'Figure'

                title = (re.sub(r'[\\/:$*?"<>|{},]', '', title)
                            .replace(' ', '_')
                            .replace('text', ''))
                plt.savefig(os.path.join(save_path, f"{title.lower()}.{suffix}"))
                log_print(logger, f'The figure \'{title}\' saved in:\n\t{save_path}\n')
            if show:
                plt.show()
            else:
                plt.close()

        return wrapper

    return decorator


class TrainingAnalyse:
    def __init__(self, train_losses=None, val_losses=None, epoch_range: tuple = None):
        self._train_loss = train_losses
        self._val_loss = val_losses
        self._epoch = np.arange(1, len(self._train_loss) + 1) if epoch_range is None else np.arange(*epoch_range)

    @save_show_wrapper()
    def loss_curve(self):
        plt.plot(self._epoch, self._train_loss, label='Training Loss')
        if self._val_loss is not None:
            plt.plot(self._epoch, self._val_loss, label='Validation Loss')


class Result:
    def __init__(self, target, pred, x):
        self._true_value = target
        self._pred_value = pred if isinstance(pred, tuple) else [pred]

        self._x = np.arange(len(self._pred_value[0])) if x is None else x

    # def __len__(self):
    #     return len(self._pred_value)

    def get_data(self):
        return self._true_value, self._pred_value


class RegressionResult(Result):
    def __init__(self, target, pred, target_label=None, pred_label=None, x=None):
        super().__init__(target=target, pred=pred, x=x)
        # self._error = self._pred_value - self._true_value
        self._error_om = np.log10(self._pred_value / self._true_value)
        self._true_label = target_label if target_label is not None else 'True Values'
        if pred_label is None:
            if len(self._pred_value) > 1:
                self._pred_label = [f'Predicted Values {i}' for i in range(len(self._pred_value))]
            else:
                self._pred_label = ['Predicted Values']
        else:
            self._pred_label = pred_label

    @staticmethod
    def _statistics(true_value, pred_value, label, logger=None):
        # Initialize metrics with None
        mse = rmse = mae = mape = msle = rmsle = r2 = evs = max_error = max_error_om =  '-'

        try:
            mse = mean_squared_error(true_value, pred_value)
            rmse = np.sqrt(mse)
        except ValueError:
            pass  # Keep the values as None if there's an error

        try:
            mae = mean_absolute_error(true_value, pred_value)
        except ValueError:
            pass

        try:
            mape = mean_absolute_percentage_error(true_value, pred_value)
        except ValueError:
            pass

        try:
            msle = mean_squared_log_error(true_value, pred_value)
            rmsle = np.sqrt(msle)
        except ValueError:
            pass

        try:
            r2 = r2_score(true_value, pred_value)
        except ValueError:
            pass

        try:
            evs = explained_variance_score(true_value, pred_value)
        except ValueError:
            pass

        error = pred_value - true_value
        error_om = np.log10(pred_value / true_value)
        if len(error) > 0:
            max_error = np.abs(error).max()
            max_error_om = np.abs(error_om).max()

        header = ['Statistics', 'Abbr.', 'Value']
        data = [
            ['Maximum Error', 'ME', max_error],
            ['Maximum OM of Error', 'MOE', max_error_om],
            ['Mean Squared Error', 'MSE', mse],
            ['Root Mean Squared Error', 'RMSE', rmse],
            ['Mean Absolute Error', 'MAE', mae],
            ['Mean Absolute Percentage Error', 'MAPE', mape],
            ['Mean Squared Logarithmic Error', 'MSLE', msle],
            ['Root Mean Squared Logarithmic Error', 'RMSLE', rmsle],
            ['Coefficient of Determination', 'R^2', r2],
            ['Explained Variance Score', 'EVS', evs]
        ]

        message = f'The Statistics Analysis of {label} in Results\n'
        message += TableFormat().generate(header=header, data=data).get_table()
        log_print(logger, message=message)

    def statistics(self, logger=None):
        for pred_value, pred_label in zip(self._pred_value, self._pred_label):
            self._statistics(self._true_value, pred_value, pred_label, logger)

    @save_show_wrapper()
    def plot(self, l: str | list = '-', alpha=1):
        ls = [l] * (len(self._pred_value) + 1) if isinstance(l, str) else l

        plt.plot(self._x, self._true_value, label=self._true_label,
                 linestyle=ls[0], alpha=alpha)

        for pred, label, l in zip(self._pred_value, self._pred_label, ls[1:]):
            plt.plot(self._x, pred, label=label, linestyle=l, alpha=alpha)

    @save_show_wrapper()
    def plot_error(self, l: str | list = '-', alpha=1):
        ls = [l] * len(self._pred_value) if isinstance(l, str) else l

        for pred, label, l in zip(self._pred_value, self._pred_label, ls):
            error = pred - self._true_value
            plt.plot(self._x, error, label=label, linestyle=l, alpha=alpha)

    @save_show_wrapper()
    def plot_error_om(self, l: str | list = '-', alpha=1):
        ls = [l] * len(self._pred_value) if isinstance(l, str) else l

        for pred, label, l_l in zip(self._pred_value, self._pred_label, ls):
            error_om = np.log10(pred / self._true_value)
            plt.plot(self._x, error_om, label=label, linestyle=l, alpha=alpha)

    @save_show_wrapper()
    def scatter_fit(self, marker='.', alpha=0.5, s=5, gap=1, l='--'):

        markers = [marker] * len(self._pred_value) if isinstance(marker, str) else l

        for pred, label, marker in zip(self._pred_value, self._pred_label, markers):
            plt.scatter(self._true_value, pred, marker=marker, alpha=alpha, s=s, label=label)

        true_value_min, true_value_max = min(self._true_value), max(self._true_value)
        plt.plot([true_value_min, true_value_max], [true_value_min, true_value_max],
                 color='red', linestyle=l, label='y=x')
        plt.plot([true_value_min * 10**gap, true_value_max], [true_value_min, true_value_max * 0.1**gap],
                 color='gray', linestyle=l)
        plt.plot([true_value_min, true_value_max * 0.1**gap], [true_value_min * 10**gap, true_value_max],
                 color='gray', linestyle=l)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')

    @save_show_wrapper()
    def distribution_hist(self, bins: int = None, alpha=0.5, om=False):
        if om:
            true_value = np.log10(self._true_value)
            pred_value = [np.log10(i) for i in self._pred_value]
        else:
            true_value = self._true_value
            pred_value = self._pred_value

        if bins is None:
            bins = 'auto'
        bins = np.histogram_bin_edges(true_value, bins=bins)

        plt.hist(true_value, bins=bins, alpha=alpha, label=self._true_label)
        for pred, label in zip(pred_value, self._pred_label):
            plt.hist(pred, bins=bins, alpha=alpha, label=label)

        plt.xlabel('Value')
        plt.ylabel('Frequency')


class MdotResult(RegressionResult):
    def __init__(self, target, pred, target_label=None, pred_label=None, x=None):
        super().__init__(target=target, pred=pred, target_label=target_label, pred_label=pred_label, x=x)

    @save_show_wrapper()
    def plot_delta_m(self, alpha=0.5):
        true_value = self._true_value
        pred_value = self._pred_value
        cum_values = np.cumsum([true_value, *pred_value], axis=1)
        labels = [self._true_label, *self._pred_label]

        # plt.fill_between(self._x, np.cumsum(true_value), alpha=alpha, label=labels[0])
        for cum_value, label in zip(cum_values, labels):
            plt.fill_between(self._x, cum_value, alpha=alpha, label=label)
