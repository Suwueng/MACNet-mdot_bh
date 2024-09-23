import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pathlib import Path

from toolbox.dataload import EllipticalGalaxyData, GalaxyData
from toolbox.createdataset import fine2coarse


def save_figure(save_path: str = None, fig_name: str = None, suffix: str = 'png'):
    if save_path:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / f"{fig_name}.{suffix}")


def log_space(x: np.ndarray):
    if x.min() <= 0:
        raise ValueError("All data should be positive!")

    log_min = np.floor(np.log10(x.min()))
    log_max = np.ceil(np.log10(x.max()))

    return int(log_min), int(log_max)


def plot_contourf(r, theta, z, figsize=(10, 12), dpi=100, log_step=300, label=None, show=True, save_path=None,
                  cbar_ax=None, is_demo=False, ax=None, levels=None):
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    x = r_mesh * np.sin(theta_mesh)
    y = r_mesh * np.cos(theta_mesh)

    if levels is None:
        log_min, log_max = log_space(z)
        levels = np.logspace(log_min, log_max, log_step)
    else:
        levels, log_min, log_max = levels

    # 创建图形
    if not is_demo:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    contourf = ax.contourf(x, y, z, cmap='Spectral_r', norm=LogNorm(), levels=levels)

    # 配置坐标轴参数
    if not is_demo:
        ax.set_xlabel('x(kpc)', fontsize=25, labelpad=0)
        ax.set_ylabel('y(kpc)', fontsize=25, labelpad=0)
    ax.tick_params(direction='out', labelsize=25, length=6, width=1.5, top=True, right=True)
    ax.axis((-0.01 * r[-1], 1.01 * r[-1], -r[-1], r[-1]))  # 坐标轴范围

    # 配置图例参数
    if cbar_ax is not None:
        cbar = plt.colorbar(contourf, cax=cbar_ax)
    else:
        cbar = plt.colorbar(contourf)
    if label:
        cbar.set_label(label, fontsize=25, labelpad=30)
    cbar.set_ticks(np.logspace(log_min, log_max, log_max - log_min + 1))
    cbar.ax.tick_params(labelsize=18, pad=5, direction='in', length=6, width=1.5)

    if not is_demo:
        # 设置坐标轴边框宽度
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)

        # 调整图形边距
        plt.tight_layout()

        if save_path:
            save_figure(save_path, f"{label}_contourf")

        if show:
            plt.show()
        else:
            plt.close()
    else:
        ax.get_xaxis().set_visible(False)  # Hide x-axis for the last subplot
        ax.get_yaxis().set_visible(False)  # Hide y-axis for the last subplot


def plot_contourf_double(r1, theta1, z1, r2, theta2, z2, figsize=(20, 12), dpi=100, log_step=300, label=None, show=True,
                         save_path=None):
    r_mesh1, theta_mesh1 = np.meshgrid(r1, theta1)
    r_mesh2, theta_mesh2 = np.meshgrid(r2, theta2)
    x1 = r_mesh1 * np.sin(theta_mesh1)
    y1 = r_mesh1 * np.cos(theta_mesh1)
    x2 = r_mesh2 * np.sin(theta_mesh2)
    y2 = r_mesh2 * np.cos(theta_mesh2)

    log_min1, log_max1 = log_space(z1)
    log_min2, log_max2 = log_space(z2)

    log_min = min(log_min1, log_min2)
    log_max = max(log_max1, log_max2)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    contourf1 = axs[0].contourf(x1, y1, z1, cmap='Spectral_r', norm=LogNorm(),
                                levels=np.logspace(log_min, log_max, log_step))
    contourf2 = axs[1].contourf(x2, y2, z2, cmap='Spectral_r', norm=LogNorm(),
                                levels=np.logspace(log_min, log_max, log_step))

    axs[0].set_xlabel('x(kpc)', fontsize=25, labelpad=0)
    axs[0].set_ylabel('y(kpc)', fontsize=25, labelpad=0)
    axs[0].tick_params(direction='out', labelsize=25, length=6, width=1.5, top=True, right=True)
    axs[0].axis((-0.01 * r1[-1], 1.01 * r1[-1], -r1[-1], r1[-1]))

    axs[1].set_xlabel('x(kpc)', fontsize=25, labelpad=0)
    axs[1].set_ylabel('y(kpc)', fontsize=25, labelpad=0)
    axs[1].tick_params(direction='out', labelsize=25, length=6, width=1.5, top=True, right=True)
    axs[1].axis((-0.01 * r2[-1], 1.01 * r2[-1], -r2[-1], r2[-1]))

    for ax in axs:
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)

    # Create an axis for the colorbar on the right of the plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contourf1, cax=cbar_ax)
    if label:
        cbar.set_label(label, fontsize=25, labelpad=30)
    cbar.set_ticks(np.logspace(log_min, log_max, log_max - log_min + 1))
    cbar.ax.tick_params(labelsize=25, pad=5, direction='in', length=6, width=1.5)

    plt.subplots_adjust(right=0.9)  # Adjust the right space to fit the colorbar

    save_figure(save_path, f"{label}_contourf_double")

    if show:
        plt.show()
    else:
        plt.close()


def demo_fine2coarse(data: GalaxyData, prop='abun_wind', filter_sizes=None):
    if filter_sizes is None:
        filter_sizes = [(2, 4), (3, 5), (4, 7), (6, 10), (8, 28), (9, 35)]

    # 初始化 log_min 和 log_max
    log_min, log_max = float('inf'), float('-inf')

    # 计算每个数据集的 log_min 和 log_max
    def update_log_range(arr):
        nonlocal log_min, log_max
        # 筛选正值
        arr = arr[arr > 0]
        if arr.size > 0:
            current_log_min, current_log_max = log_space(arr)
            log_min = min(log_min, current_log_min)
            log_max = max(log_max, current_log_max)

    # 计算初始数据的 log_min 和 log_max
    update_log_range(getattr(data.galaxy_prop, prop))

    # 计算所有滤波后的数据的 log_min 和 log_max
    for filter_size in filter_sizes:
        tmp = fine2coarse(data, filter_size[0], filter_size[1])
        update_log_range(getattr(tmp.galaxy_prop, prop))

    levels = np.logspace(log_min, log_max, 300)

    # 创建子图和共享色条
    fig, axs = plt.subplots(1, len(filter_sizes) + 1, figsize=(22, 6), dpi=100,
                            gridspec_kw={'width_ratios': [2] * (len(filter_sizes)+1)})
    cbar_ax = fig.add_axes([0.93, 0.025, 0.02, 0.95])  # 色条位置

    # 绘制初始图像
    plot_contourf(data.r, data.theta, getattr(data.galaxy_prop, prop),
                  cbar_ax=cbar_ax, label=prop, is_demo=True, ax=axs[0], levels=(levels, log_min, log_max))

    # 绘制过滤后的图像
    for idx, filter_size in enumerate(filter_sizes):
        tmp = fine2coarse(data, filter_size[0], filter_size[1])
        plot_contourf(tmp.r, tmp.theta, getattr(tmp.galaxy_prop, prop),
                      cbar_ax=cbar_ax, label=None, is_demo=True, ax=axs[idx + 1], levels=(levels, log_min, log_max))

    plt.tight_layout()
    plt.subplots_adjust(right=0.92)  # 调整右侧边距以适应色条

    plt.show()


if __name__ == "__main__":
    data = EllipticalGalaxyData('/home/peng/dataset/Simulation_BH/disk_galaxy/fiducial', 400)
    plot_contourf(data.r, data.theta, data.galaxy_prop.abun_wind, label='abun_wind')
