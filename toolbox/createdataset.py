import multiprocessing
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # Loading the environment variables in the .env file
database_path = os.getenv("DATABASE_PATH")

from toolbox.dataload import EllipticalGalaxyData, DiskGalaxyData, GalaxyData, read_pickle


def eg_fine_dataset(folder_path=None, dumps_list=None, save_path=None, suffix=None, time_lag=None, m_bh=None):
    """
    Read elliptical galaxy hdf format data, dump to pkl format data.

    :param folder_path: The path of folder stored data
    :param dumps_list: List of data numbers
    :param save_path: Save path of pkl format data
    :param suffix: Data file format, support 'hdf5' (.h5) and 'pickle' (.pkl)
    :param time_lag: The time interval between adjacent time frames
    :param m_bh: The mass of black hole in the galaxy
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    match suffix:
        case None | 'h5':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Fine DataSet"):
                tmp = EllipticalGalaxyData(folder_path, dump, time_lag, m_bh)
                tmp.to_hdf5(os.path.join(save_path, f'{dump:05d}.h5'))
        case 'pkl':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Fine DataSet"):
                tmp = EllipticalGalaxyData(folder_path, dump, time_lag, m_bh)
                tmp.to_pickle(os.path.join(save_path, f'{dump:05d}.pkl'))
        case _:
            raise ValueError('Currently only hdf5(.h5) and pickle(.pkl) formats are supported.')


def dg_fine_dataset(folder_path=None, dumps_list=None, save_path=None, suffix=None, time_lag=None, m_bh=None):
    """
    Read disk galaxy hdf format data, dump to pkl format data.

    :param folder_path: The path of folder stored data
    :param dumps_list: List of data numbers
    :param save_path: Save path of pkl format data
    :param suffix: Data file format, support 'hdf5' (.h5) and 'pickle' (.pkl)
    :param time_lag: The time interval between adjacent time frames
    :param m_bh: The mass of black hole in the galaxy
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    match suffix:
        case None | 'h5':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Fine DataSet"):
                tmp = DiskGalaxyData(folder_path, dump, time_lag, m_bh)
                tmp.to_hdf5(os.path.join(save_path, f'{dump:05d}.h5'))
        case 'pkl':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Fine DataSet"):
                tmp = DiskGalaxyData(folder_path, dump, time_lag, m_bh)
                tmp.to_pickle(os.path.join(save_path, f'{dump:05d}.pkl'))
        case _:
            raise ValueError('Currently only hdf5(.h5) and pickle(.pkl) formats are supported.')


def coarse_dataset(folder_path, dumps_list=None, theta_scale=9, r_scale=35, save_path=None, suffix=None):
    """
    Read the fine mesh data, convert it to the coarse mesh data and save it.

    :param folder_path: The path of folder stored data
    :param dumps_list: List of data numbers
    :param theta_scale: Theta direction dimension of the conversion operator
    :param r_scale: R direction dimension of the conversion operator
    :param save_path: Save path of the coarse mesh data
    :param suffix: Data file format, support 'hdf5' (.h5) and 'pickle' (.pkl)
    :param max_workers: Maximum number of threads to use
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    match suffix:
        case None | 'h5':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Coarse DataSet"):
                galaxy = GalaxyData().read_hdf5(os.path.join(folder_path, f'{dump:05d}.h5'))
                fine2coarse(galaxy, theta_scale, r_scale).to_hdf5(os.path.join(save_path, f'{dump:05d}.h5'))
        case 'pkl':
            for dump in tqdm(dumps_list, total=len(dumps_list), desc="Creating Coarse DataSet"):
                galaxy = read_pickle(os.path.join(folder_path, f'{dump:05d}.pkl'))
                fine2coarse(galaxy, theta_scale, r_scale).to_pickle(os.path.join(save_path, f'{dump:05d}.pkl'))
        case _:
            raise ValueError('Currently only hdf5(.h5) and pickle(.pkl) formats are supported.')
        

def fine2coarse(data: GalaxyData, theta_scale=None, r_scale=None):
    """
    Reduce resolution that convert a fine mesh to a coarse mesh
    
    :param data: Fine mesh data
    :param theta_scale: The dimensions of the window in the theta direction
    :param r_scale: the dimensions of the window in the r direction
    :return: 
    """
    data_coarse = GalaxyData()
    theta = data.theta
    r = data.r
    theta_step, r_step = theta_scale, r_scale

    data_coarse.time = data.time
    data_coarse.file_name = data.file_name
    data_coarse.m_bh = data.m_bh
    mass = data.mass

    # Initial
    gp_list = list(data.galaxy_prop.__dict__)
    shape = (len(theta) // theta_step, len(r) // r_step)
    for gp in gp_list:
        setattr(data_coarse.galaxy_prop, gp, np.zeros(shape))
    data_coarse.volume = np.zeros(shape)
    data_coarse.mass = np.zeros(shape)

    for i_new, i in enumerate(range(0, len(theta), theta_step)):
        for j_new, j in enumerate(range(0, len(r), r_step)):
            theta_slice = slice(i, i + theta_scale)
            r_slice = slice(j, j + r_scale)

            data_coarse.galaxy_prop.abun_init[i_new, j_new] = (
                np.average(data.galaxy_prop.abun_init[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.abun_ism[i_new, j_new] = (
                np.average(data.galaxy_prop.abun_ism[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.abun_jet[i_new, j_new] = (
                np.average(data.galaxy_prop.abun_jet[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.abun_wind[i_new, j_new] = (
                np.average(data.galaxy_prop.abun_wind[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.dnewstar[i_new, j_new] = (
                np.average(data.galaxy_prop.dnewstar[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.v1[i_new, j_new] = (
                np.average(data.galaxy_prop.v1[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.v2[i_new, j_new] = (
                np.average(data.galaxy_prop.v2[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.v3[i_new, j_new] = (
                np.average(data.galaxy_prop.v3[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))
            data_coarse.galaxy_prop.gas_energy[i_new, j_new] = (
                np.average(data.galaxy_prop.gas_energy[theta_slice, r_slice], weights=mass[theta_slice, r_slice]))

            data_coarse.galaxy_prop.density[i_new, j_new] = (np.sum(mass[theta_slice, r_slice]) /
                                                             np.sum(data.volume[theta_slice, r_slice]))
            data_coarse.volume[i_new, j_new] = np.sum(data.volume[theta_slice, r_slice])
            data_coarse.mass[i_new, j_new] = np.sum(data.mass[theta_slice, r_slice])

    data_coarse.galaxy_prop.temperature = 93 * data_coarse.galaxy_prop.gas_energy / data_coarse.galaxy_prop.density

    data_coarse.theta = theta[[i for i in range(int(theta_scale / 2), len(theta), theta_step)]]
    data_coarse.r = r[[i for i in range(int(r_scale / 2), len(r), r_step)]]
    data_coarse.lentheta = len(data_coarse.theta)
    data_coarse.lenr = len(data_coarse.r)
    data_coarse.dr = np.zeros(data_coarse.lenr)
    data_coarse.dtheta = np.zeros(data_coarse.lentheta)
    for i in range(data_coarse.lenr - 1):
        data_coarse.dr[i] = data_coarse.r[i + 1] - data_coarse.r[i]
    data_coarse.dr[-1] = data_coarse.dr[-2]
    for i in range(data_coarse.lentheta):
        data_coarse.dtheta[i] = 9 / 10 * np.pi / data_coarse.lentheta

    return data_coarse


def encapsulate_data(folder_path: str, frame_num: int, suffix='h5', y_pattern='*.csv',
                     r_acc=None, gamma=None, gamma1=None, w=None):
    """
    The independent and dependent variables as well as the calculated Bondi accretion rate data
    were encapsulated as.npy format files.

    :param folder_path: The path of folder stored data
    :param frame_num: The number of time frames in each sample
    :param suffix: Suffix of the independent variable file
    :param y_pattern: The pattern of matching the dependent variable file
    :param r_acc: List of accretion radius
    :param gamma: Polytropic index
    :param gamma1: a parameter
    :param w: The weighting method for calculating the density and the speed of light at infinity
    :return:
    """

    # Encapsulate x and Bondi accretion
    encapsulated_x = []
    encapsulated_mdot_bondi = []
    ttmp = []
    for data_path in sorted(glob(os.path.join(folder_path, f'*.{suffix}'))):
        tmp = []
        match suffix:
            case 'h5':
                data = GalaxyData().read_hdf5(data_path)
            case 'pkl':
                data = read_pickle(data_path)
            case _:
                raise ValueError('Currently only hdf5(.h5) and pickle(.pkl) formats are supported.')

        tmp += [getattr(data.galaxy_prop, i) for i in list(data.galaxy_prop.__dict__)]
        tmp.append(data.volume)
        tmp.append(data.mass)
        tmp += np.meshgrid(data.r, data.theta)

        ttmp.append(tmp)
        encapsulated_mdot_bondi.append([data.bondi_acc(r, gamma, gamma1, w) for r in r_acc])

    for num in range(len(ttmp) - frame_num + 1):
        encapsulated_x.append([ttmp[num + i] for i in range(frame_num)])
    encapsulated_mdot_bondi = encapsulated_mdot_bondi[(frame_num - 1):]

    encapsulated_x = np.array(encapsulated_x)  # (sample，frame，channel, row, column)
    encapsulated_mdot_bondi = np.array(encapsulated_mdot_bondi)  # (sample, r_acc)
    print(folder_path, len(encapsulated_x))

    # Encapsulate y
    encapsulated_y = []
    for file_path in sorted(glob(os.path.join(os.path.dirname(os.path.dirname(folder_path)), y_pattern))):
        encapsulated_y += list(pd.read_csv(file_path)['value'])[frame_num - 1:]
        print(file_path, len(list(pd.read_csv(file_path)['value'])[frame_num - 1:]))

    encapsulated_y = np.array(encapsulated_y)

    return encapsulated_x, encapsulated_y, encapsulated_mdot_bondi
