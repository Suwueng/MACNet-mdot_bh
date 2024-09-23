import os
import h5py
import pickle
import warnings

import numpy as np
import pandas as pd
from pyhdf.SD import SD
from scipy.interpolate import griddata

import toolbox

gp_list = ['v1',
           'v2',
           'v3',
           'density',
           'gas_energy',
           'dnewstar',
           'abun_ism',
           'abun_wind',
           'abun_init',
           'abun_jet',
           'temperature']


class GalaxyData:
    """
    The parent class of reading hdf files that stores data about various types of galaxies.
    """
    def __init__(self, file_name=None, dump=None, time_lag=None, m_bh=None):
        self.galaxy_prop = GalaxyProperty()
        if file_name is not None and dump is not None:
            time_lag = 0.0125 if time_lag is None else time_lag

            self.time = dump * time_lag
            self.m_bh = m_bh

            self.file_name = os.path.join(file_name, f"hdfra.{dump:05d}")
            hdf_file = SD(self.file_name)

            self.galaxy_prop.v1 = hdf_file.select("Data-Set-2")[0, :, :]
            self.galaxy_prop.v2 = hdf_file.select("Data-Set-3")[0, :, :]
            self.galaxy_prop.v3 = hdf_file.select("Data-Set-4")[0, :, :]

            self.galaxy_prop.density = hdf_file.select("Data-Set-5")[0, :, :]

            self.r = hdf_file.select("fakeDim2")[:]
            self.theta = hdf_file.select("fakeDim1")[:]
            self.lentheta = len(self.theta)
            self.lenr = len(self.r)
            hdf_file.end()

            self.dr = np.diff(self.r, prepend=0)
            self.dtheta = np.full(self.lentheta, 9 / 10 * np.pi / self.lentheta)

            self.volume = 1/2 * np.outer(self.dtheta, np.diff(np.square(self.r), prepend=0))
            self.mass = self.galaxy_prop.density * self.volume
            self.r_bondi = None


    def uniform(self, new):
        """
        The data and target data are unified by the interpolation in the shape and the position of the calculation node.

        ** Note: This method should be used with caution as it can produce negative values in non-negative data. **

        :param new: The target data
        :return:
        """
        for p in gp_list:
            Z_new = interpolation(self.r, self.theta, getattr(self.galaxy_prop, p), new.r, new.theta)
            setattr(self.galaxy_prop, p, Z_new)

        for attr in ['r', 'theta', 'lenr', 'lentheta', 'dr', 'dtheta', 'volume']:
            setattr(self, attr, getattr(new, attr))

        return self

    def to_pickle(self, file_path):
        """
        Save the hdf file data as .pkl format.

        :param file_path: Save path
        :return:
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def to_hdf5(self, file_path):
        with h5py.File(file_path, 'w') as f:
            f.attrs['file_name'] = self.file_name
            f.attrs['m_bh'] = self.m_bh
            f.attrs['time'] = self.time

            hdfra = f.create_group('hdfra')
            hdfra.create_dataset('dr', data=self.dr)
            hdfra.create_dataset('dtheta', data=self.dtheta)
            hdfra.create_dataset('mass', data=self.mass)
            hdfra.create_dataset('r', data=self.r)
            hdfra.create_dataset('theta', data=self.theta)
            hdfra.create_dataset('volume', data=self.volume)

            galaxy_prop = hdfra.create_group('galaxy_prop')
            galaxy_prop.create_dataset('abun_init', data=self.galaxy_prop.abun_init)
            galaxy_prop.create_dataset('abun_ism', data=self.galaxy_prop.abun_ism)
            galaxy_prop.create_dataset('abun_jet', data=self.galaxy_prop.abun_jet)
            galaxy_prop.create_dataset('abun_wind', data=self.galaxy_prop.abun_wind)
            galaxy_prop.create_dataset('density', data=self.galaxy_prop.density)
            galaxy_prop.create_dataset('dnewstar', data=self.galaxy_prop.dnewstar)
            galaxy_prop.create_dataset('gas_energy', data=self.galaxy_prop.gas_energy)
            galaxy_prop.create_dataset('temperature', data=self.galaxy_prop.temperature)
            galaxy_prop.create_dataset('v1', data=self.galaxy_prop.v1)
            galaxy_prop.create_dataset('v2', data=self.galaxy_prop.v2)
            galaxy_prop.create_dataset('v3', data=self.galaxy_prop.v3)

    def read_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            hdfra = f['hdfra']
            self.dr = hdfra['dr'][:]
            self.dtheta = hdfra['dtheta'][:]
            self.mass = hdfra['mass'][:]
            self.r = hdfra['r'][:]
            self.theta = hdfra['theta'][:]
            self.volume = hdfra['volume'][:]

            self.lenr = len(self.r)
            self.lentheta = len(self.theta)

            self.file_name = f.attrs['file_name']
            self.m_bh = f.attrs['m_bh']
            self.time = f.attrs['time']

            galaxy_prop = hdfra['galaxy_prop']
            self.galaxy_prop.abun_init = galaxy_prop['abun_init'][:]
            self.galaxy_prop.abun_ism = galaxy_prop['abun_ism'][:]
            self.galaxy_prop.abun_jet = galaxy_prop['abun_jet'][:]
            self.galaxy_prop.abun_wind = galaxy_prop['abun_wind'][:]
            self.galaxy_prop.density = galaxy_prop['density'][:]
            self.galaxy_prop.dnewstar = galaxy_prop['dnewstar'][:]
            self.galaxy_prop.gas_energy = galaxy_prop['gas_energy'][:]
            self.galaxy_prop.temperature = galaxy_prop['temperature'][:]
            self.galaxy_prop.v1 = galaxy_prop['v1'][:]
            self.galaxy_prop.v2 = galaxy_prop['v2'][:]
            self.galaxy_prop.v3 = galaxy_prop['v3'][:]

        return self


    def bondi_acc(self, r_acc, gamma=None, gamma1=None, w=None):
        """
        Calculate the Bondi-Hoyle accretion rate base on the radius of accretion.

        :param r_acc: The radius of accretion
        :param gamma: Polytropic index
        :param gamma1: a parameter
        :param w: The weighting method for calculating the density and the speed of light at infinity
        :return: The Bondi-Hoyle accretion rate
        """
        # 运行前检查
        if not np.any(self.r < r_acc):
            raise RuntimeError(f"There is no computation point within the accretion radius ({r_acc}).")

        G = 112
        pi = np.pi
        gamma = 5 / 3 if gamma is None else gamma
        gamma1 = 5 / 3 - 1 if gamma1 is None else gamma1

        e = self.galaxy_prop.gas_energy[:, self.r < r_acc]
        d = self.galaxy_prop.density[:, self.r < r_acc]
        v = self.volume[:, self.r < r_acc]
        m = d * v
        if w is None or w == 'm':       # Default weighting method is mass
            w = m
        elif w == 'v':
            w = v

        d_inf = np.average(d, weights=w)
        c_s_inf = np.average(np.sqrt(e * gamma * gamma1), weights=w)

        lambda_c = 1/4 * (2/(5-3*gamma))**((5-3*gamma) / 2*(gamma-1)) if gamma != 5/3 else 1
        m_dot_bondi = (4 * pi * lambda_c * G**2 * self.m_bh**2 * d_inf) / c_s_inf**3

        return m_dot_bondi


class EllipticalGalaxyData(GalaxyData):
    """
    The class is loading elliptical galaxy data.
    """
    def __init__(self, file_name, dump, time_lag=None, m_bh=None):
        time_lag = 0.0001 if time_lag is None else time_lag
        m_bh = 4.5e9 / 2.5e7 if m_bh is None else m_bh
        super().__init__(file_name, dump, time_lag, m_bh)

        hdf_file = SD(self.file_name)

        self.galaxy_prop.gas_energy = hdf_file.select("Data-Set-6")[0, :, :]
        self.galaxy_prop.dnewstar = hdf_file.select("Data-Set-8")[0, :, :]

        self.galaxy_prop.abun_ism = hdf_file.select("Data-Set-9")[0, :, :]
        self.galaxy_prop.abun_wind = hdf_file.select("Data-Set-10")[0, :, :]
        self.galaxy_prop.abun_init = hdf_file.select("Data-Set-12")[0, :, :]
        self.galaxy_prop.abun_jet = hdf_file.select("Data-Set-13")[0, :, :]
        self.galaxy_prop.temperature = 93 * self.galaxy_prop.gas_energy / self.galaxy_prop.density

        hdf_file.end()


class DiskGalaxyData(GalaxyData):
    """
    The class is loading disk galaxy data.
    """
    def __init__(self, file_name, dump, time_lag=None, m_bh=None):
        time_lag = 0.0125 if time_lag is None else time_lag
        m_bh = 5e7 / 2.5e7 if m_bh is None else m_bh
        super().__init__(file_name, dump, time_lag, m_bh)

        hdf_file = SD(str(self.file_name))

        self.galaxy_prop.gas_energy = hdf_file.select("Data-Set-10")[0, :, :]
        self.galaxy_prop.dnewstar = hdf_file.select("Data-Set-13")[0, :, :]

        self.galaxy_prop.abun_ism = hdf_file.select("Data-Set-18")[0, :, :]
        self.galaxy_prop.abun_wind = hdf_file.select("Data-Set-19")[0, :, :]
        self.galaxy_prop.abun_init = hdf_file.select("Data-Set-23")[0, :, :]
        self.galaxy_prop.abun_jet = hdf_file.select("Data-Set-20")[0, :, :]
        self.galaxy_prop.temperature = 93 * self.galaxy_prop.gas_energy / self.galaxy_prop.density

        hdf_file.end()


class EllipticalGalaxyMdot:
    """
    Process the accretion rate data calculated in the simulation.
    """
    def __init__(self, time_lag=None):
        self.data = None
        self.time = None
        self.mdot_bh = None
        self.scope = None
        self.interval = 0.0001 if time_lag is None else time_lag

    def load(self, file_path):
        """
        Read the data file stored the accretion rate.

        :param file_path: Data file path
        :return:
        """
        tmp = pd.read_csv(file_path, sep='\s+', header=None)
        tmp.columns = ['time', 'mbh', 'mdot_bh/mdot_edd', 'mdot_edd', 'mdot_bondi', 'hot_wind_power', 'jet_power',
                       'cold_wind_power', 'max(glob_sfr_tot,1.scope-99)', 'max(cold_tot,1.scope-99)']

        self.data = pd.DataFrame(
            {
                'time': tmp['time'],
                'mdot_bh': tmp['mdot_bh/mdot_edd'] * tmp['mdot_edd']
            }
        )

        return self

    def __alignment(self, dumps_list):
        """
        Extract the target range data from the data file.

        :param dumps_list: The target range list
        :return:
        """
        range_start, range_end = dumps_list[0] * self.interval, dumps_list[-1] * self.interval

        mask = (self.data['time'] >= range_start) & (self.data['time'] <= range_end)

        self.time = self.data[mask]['time']
        self.mdot_bh = self.data[mask]['mdot_bh']

    def __average_mdot_bh(self):
        """
        The accretion rate value at the target time frame was obtained by averaging the data within the allowed scope.

        :return:
        """
        tmp = pd.DataFrame({'time': self.time, 'mdot_bh': self.mdot_bh})

        tmp['rounded_time'] = (tmp['time'] / self.interval).round() * self.interval

        mask = ((tmp['time'] >= tmp['rounded_time'] - self.scope) &
                (tmp['time'] <= tmp['rounded_time'] + self.scope))
        tmp = tmp[mask]

        tmp_avg = tmp.groupby('rounded_time', as_index=False).mean()

        self.time = tmp_avg['rounded_time']
        self.mdot_bh = tmp_avg['mdot_bh']

    def process(self, dumps_list, scope=None):
        """
        Extract the accretion rate of target time frames in the data file.

        :param scope: The scope of allowed time frames
        :param dumps_list: The target range list
        :return: The accretion rate of target time frames
        """
        self.scope = 0.00005 if scope is None else scope

        self.__alignment(dumps_list)
        self.__average_mdot_bh()

        if len(self.time) < len(dumps_list):
            warnings.warn(f"Some time frames have no corresponding accretion rate. "
                          f"It is recommended to increase the scope ({scope}) ", UserWarning)
        if self.interval < scope:
            warnings.warn(f"Scope ({scope}) is greater than self.interval ({self.interval}). "
                          f"It is recommended to increase the scope ({scope}) ", UserWarning)

        return pd.DataFrame({'time': self.time, 'value': self.mdot_bh})


class DiskGalaxyMdot(EllipticalGalaxyMdot):
    def __init__(self, time_lag=None):
        time_lag = 0.0125 if time_lag is None else time_lag
        super().__init__(time_lag=time_lag)

    def load(self, file_path):
        with open(file_path) as file:
            lines = file.readlines()

        time_tmp = []
        mdot_bh_tmp = []

        for line in lines:
            equal_index = line.find('=')
            if equal_index != -1:
                right_side = line[equal_index + 1:].strip().split()
                right_side[1] = right_side[1].replace("D", "E")
                right_side[6] = right_side[6].replace("D", "E")
                time_tmp.append(float(right_side[1]))
                mdot_bh_tmp.append(float(right_side[6]))

        self.data = pd.DataFrame({'time': time_tmp, 'mdot_bh': mdot_bh_tmp})

        return self


def interpolation(x, y, z, x_new, y_new, method='cubic'):
    """
    The size of the data is transformed by the interpolation.

    :param x: X-axis scale
    :param y: Y-axis scale
    :param z: Target data
    :param x_new: Target X-axis scale
    :param y_new: Target Y-axis scale
    :param method: Method of interpolation
    :return: Target size data
    """
    x, y = np.meshgrid(x, y)
    x_new, y_new = np.meshgrid(x_new, y_new)

    points = np.array([x.ravel(), y.ravel()]).T
    values = z.ravel()

    z_new = griddata(points, values, (x_new, y_new), method=method)
    z_new[np.isnan(z_new)] = griddata(points, values, (x_new, y_new), method='nearest')[np.isnan(z_new)]

    return z_new


class GalaxyProperty:
    """
    The properties of galaxy.
    """
    def __init__(self):
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.density = None
        self.gas_energy = None
        self.dnewstar = None
        self.abun_ism = None
        self.abun_wind = None
        self.abun_init = None
        self.abun_jet = None
        self.temperature = None


def read_pickle(file_path):
    """
    Read data in .pkl format.

    :param file_path: Path of target .pkl file
    :return:
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# if __name__ == '__main__':
#     eg = EllipticalGalaxyData(
#         file_name=os.path.join(toolbox.database_path, 'elliptical_galaxy/elliptical_galaxy_21999'),
#         dump=11728,
#         time_lag=1e-4,
#         m_bh=180
#     )
#
#     eg.to_hdf5(os.path.join(toolbox.database_path, '11728.h5'))
#
#     eg1 = GalaxyData().read_hdf5('/home/peng/MyFile/DataSet/Simulation_BH/11728.h5')
