import os
import h5py
import warnings

import numpy as np
import pandas as pd

from pyhdf.SD import SD
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Union, Type

# ============= Utility Functions =============


def process_files_in_parallel(files: List[str], process_fn, max_workers: int = -1):
    """
    Process files in parallel using the given processing function.
    """
    max_workers = os.cpu_count() if max_workers == -1 else min(os.cpu_count(), max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fn, files), total=len(files), desc="Processing files"))
    return results


def ensure_dir_exists(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def extract_dump(file_path: str) -> int:
    """
    Extract the dump number from the file name. Raise ValueError if extraction fails.
    """
    try:
        return int(os.path.basename(file_path).split('.')[-1])
    except ValueError:
        raise ValueError(f"Invalid file format: {file_path}.")


# ============= Data Load =============


class GridData:
    """
    Properties of a galaxy grid.
    """

    def __init__(self) -> None:
        self.initialize_attributes()

    def initialize_attributes(self) -> None:
        attributes = [
            "v1", "v2", "v3", "density", "gas_energy", "dnewstar", "temperature", "volume", "mass", "r", "theta",
            "abun_ism", "abun_wind", "abun_init", "abun_jet"
        ]
        for attr in attributes:
            setattr(self, attr, np.array([]))

    @property
    def attributes(self) -> list[str]:
        return list(self.__dict__.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        return self.r.shape


class GalaxyData:
    """
    Base class for handling HDF data for galaxies.
    """

    def __init__(self) -> None:
        self.initialize_attributes()

    def initialize_attributes(self) -> None:
        self.time = None

        self.galaxy_prop = GridData()

        self.mbh = None
        self.mdot_macer = None
        self.mdot_edd = None
        self.mdot_bondi = None

        self.r_vector = None
        self.theta_vector = None

        self.file_path = None
        self.logfile_path = None

    @property
    def dr(self) -> np.ndarray:
        self._validate_nonempty()
        return np.diff(self.r_vector, prepend=0)

    @property
    def dtheta(self) -> np.ndarray:
        self._validate_nonempty()
        return np.full(len(self.theta_vector), 0.9 * np.pi / len(self.theta_vector))

    @property
    def shape(self) -> tuple:
        return self.galaxy_prop.shape

    def to_hdf5(self, save_path: str) -> None:
        """
        Save the galaxy data to an HDF5 file.
        """
        self._validate_save_path(save_path)

        try:
            with h5py.File(save_path, 'w') as f:
                f.attrs.update({
                    'time': self.time,
                    'mbh': self.mbh,
                    'mdot_macer': self.mdot_macer,
                    'mdot_edd': self.mdot_edd,
                    'mdot_bondi': self.mdot_bondi,
                    'file_path': self.file_path,
                    'logfile_path': self.logfile_path,
                })

                hdfra = f.create_group('hdfra')
                for attr in ["r_vector", "theta_vector"]:
                    hdfra.create_dataset(attr, data=getattr(self, attr))

                galaxy_prop_group = hdfra.create_group('galaxy_prop')
                for attr in self.galaxy_prop.attributes:
                    galaxy_prop_group.create_dataset(attr, data=getattr(self.galaxy_prop, attr))
        except Exception as e:
            raise IOError(f"Error while saving to HDF5: {e}")

    def read_hdf5(self, file_path: str) -> "GalaxyData":
        """
        Read the galaxy data from an HDF5 file.
        """
        self._validate_file_path(file_path, "HDF5 file")

        try:
            with h5py.File(file_path, 'r') as f:
                for attr in ['time', 'mbh', 'mdot_macer', 'mdot_edd', 'mdot_bondi', 'file_path', 'logfile_path']:
                    setattr(self, attr, f.attrs.get(attr))

                hdfra = f['hdfra']
                for attr in ["r_vector", "theta_vector"]:
                    setattr(self, attr, hdfra[attr][:])

                galaxy_prop_group = hdfra['galaxy_prop']
                for attr in self.galaxy_prop.attributes:
                    if attr in galaxy_prop_group:
                        setattr(self.galaxy_prop, attr, galaxy_prop_group[attr][:])
                    else:
                        warnings.warn(f"Missing attribute in HDF5: {attr}")
        except Exception as e:
            raise IOError(f"Error while reading from HDF5: {e}")

        return self

    def read_rawfile(self, file_path: str, dump: int, logfile_path: str, time_lag: float, scope: float,
                     datasets: dict) -> "GalaxyData":
        """
        General method to load galaxy data from HDF and log files.
        """
        self.time = dump * time_lag
        self.file_path = os.path.join(file_path, f"hdfra.{dump:05d}")
        self.logfile_path = logfile_path

        # Read the HDF file
        self._read_hdffile(datasets)
        # Read the log file
        self._read_logfile(self.logfile_path, scope)
        # Calculate the Bondi accretion rate
        self._calculate_mdot_bondi()

        return self

    def _calculate_mdot_bondi(self, r_acc: float = 1.0, gamma: float = 5 / 3) -> np.float64:
        if self.mbh is None:
            raise ValueError("Black hole mass (mbh) must be set before calculating Bondi accretion rate.")
        G = 112
        pi = np.pi
        gamma1 = gamma - 1
        lambda_c = 1 if gamma == 5 / 3 else 1 / 4 * (2 / (5 - 3 * gamma))**((5 - 3 * gamma) / (2 * (gamma - 1)))

        mask = self.r_vector < r_acc
        e = self.galaxy_prop.gas_energy[:, mask]
        d = self.galaxy_prop.density[:, mask]
        m = self.galaxy_prop.mass[:, mask]

        d_inf = np.average(d, weights=m)
        c_s_inf = np.average(np.sqrt(e / d * gamma * gamma1), weights=m)

        self.mdot_bondi = ((4 * pi * lambda_c * G**2 * self.mbh**2 * d_inf) / c_s_inf**3) / self.mdot_edd
        return self.mdot_bondi

    def _validate_file_path(self, path: str, name: str) -> None:
        if not isinstance(path, str):
            raise ValueError(f"{name} must be a string.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")

    @staticmethod
    def _validate_save_path(path: str) -> None:
        if not isinstance(path, str):
            raise ValueError("Save path must be a string.")

    def _common_dataset_setup(self, hdfra: SD) -> None:
        self.r_vector = hdfra.select("fakeDim2")[:]
        self.theta_vector = hdfra.select("fakeDim1")[:]

        datasets = {'v1': "Data-Set-2", 'v2': "Data-Set-3", 'v3': "Data-Set-4", 'density': "Data-Set-5"}
        for attr, dataset in datasets.items():
            setattr(self.galaxy_prop, attr, hdfra.select(dataset)[0, :, :])

        self.galaxy_prop.r, self.galaxy_prop.theta = np.meshgrid(self.r_vector, self.theta_vector, indexing='ij')
        self.galaxy_prop.volume = 0.5 * np.outer(self.dtheta, np.diff(np.square(self.r_vector), prepend=0))
        self.galaxy_prop.mass = self.galaxy_prop.density * self.galaxy_prop.volume

    def _read_hdffile(self, datasets: dict) -> None:
        hdfra = SD(self.file_path)
        self._common_dataset_setup(hdfra)

        # Load specific datasets
        for attr, dataset in datasets.items():
            setattr(self.galaxy_prop, attr, hdfra.select(dataset)[0, :, :])

        # Calculate temperature
        self.galaxy_prop.temperature = 93 * self.galaxy_prop.gas_energy / self.galaxy_prop.density
        hdfra.end()

    def _read_logfile(self, logfile_path: str, scope: float) -> None:
        self._parse_logfile(logfile_path, scope)

    def _parse_logfile(self, logfile_path: str, scope: float) -> None:
        """
        Abstract method for parsing log files. Subclasses must implement this.
        """
        raise NotImplementedError

    def _validate_nonempty(self) -> None:
        if self.r_vector is None or self.theta_vector is None:
            raise ValueError("Grid dimensions (r, theta) must be initialized.")

    @staticmethod
    def _scale_tranform(data, target_shape, method="sum", weights=None) -> np.ndarray:
        reshaped = data.reshape(*target_shape)
        if method == "sum":
            return reshaped.sum(axis=(1, 3))
        elif method == "average":
            if weights is None:
                raise ValueError("Weights must be provided for weighted average.")
            return np.average(reshaped, axis=(1, 3), weights=weights)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")


class EllipticalGalaxy(GalaxyData):
    """
    The class for reading elliptical galaxy data.
    """

    def _parse_logfile(self, logfile_path: str, scope: float) -> None:
        log_df = pd.read_csv(logfile_path, sep='\s+', header=None, engine='python')
        log_df.columns = [
            'time', 'mbh', 'mdot_bh/mdot_edd', 'mdot_edd', 'mdot_bondi', 'hot_wind_power', 'jet_power',
            'cold_wind_power', 'max(glob_sfr_tot,1.e-99)', 'max(cold_tot,1.e-99)'
        ]

        time_mask = (log_df['time'] >= self.time - scope) & (log_df['time'] <= self.time + scope)
        self.mbh = log_df.loc[time_mask, 'mbh'].mean()
        self.mdot_macer = log_df.loc[time_mask, 'mdot_bh/mdot_edd'].mean()
        self.mdot_edd = log_df.loc[time_mask, 'mdot_edd'].mean()

    def read_rawfile(self, file_path: str, dump: int, logfile_path: str, time_lag: float = 1e-4,
                     scope: float = 5e-5) -> "EllipticalGalaxy":
        datasets = {
            'gas_energy': "Data-Set-6",
            'dnewstar': "Data-Set-8",
            'abun_ism': "Data-Set-9",
            'abun_wind': "Data-Set-10",
            'abun_init': "Data-Set-12",
            'abun_jet': "Data-Set-13",
        }
        return super().read_rawfile(file_path, dump, logfile_path, time_lag, scope, datasets)


class DiskGalaxy(GalaxyData):
    """
    The class for reading disk galaxy data.
    """

    def _parse_logfile(self, logfile_path: str, scope: float) -> None:
        log_df = pd.read_csv(logfile_path, sep="=", usecols=[1], names=["data"], skipinitialspace=True)
        log_df = log_df["data"].str.replace("D", "E").str.split(expand=True).astype(float)
        log_df.columns = [
            "nhy", "time", "dt", "mdot", "bondi_r", "mbh", "mdot_bh", "L", "NEWstar", "dtviscmin", "cold", "hot", "jet"
        ]

        time_mask = (log_df['time'] >= self.time - scope) & (log_df['time'] <= self.time + scope)
        self.mbh = log_df.loc[time_mask, 'mbh'].mean()
        self.mdot_edd = 22 * self.mbh
        self.mdot_macer = log_df.loc[time_mask, 'mdot_bh'].mean() / self.mdot_edd

    def read_rawfile(self, file_path: str, dump: int, logfile_path: str, time_lag: float = 0.0125,
                     scope: float = 5e-4) -> "DiskGalaxy":
        datasets = {
            'gas_energy': "Data-Set-10",
            'dnewstar': "Data-Set-13",
            'abun_ism': "Data-Set-18",
            'abun_wind': "Data-Set-19",
            'abun_init': "Data-Set-23",
            'abun_jet': "Data-Set-20",
        }
        return super().read_rawfile(file_path, dump, logfile_path, time_lag, scope, datasets)


# ============= DataSet Create and Process =============


class GalaxyDataSet:
    """
    Creating and processing galaxy datasets.
    """

    def __init__(self) -> None:
        from typing import List
        self.dataset: List[GalaxyData] = []

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        self._validate_nonempty()
        first_frame = self.dataset[0]
        return (len(self.dataset), len(first_frame.galaxy_prop.attributes), *first_frame.shape)

    def read_rawfile(self, file_path: str, logfile_path: str, mode: str, time_lag: float, scope: float,
                     max_workers: int = -1) -> "GalaxyDataSet":
        galaxy_cls, time_lag, scope = self._initialize_para_set(mode, time_lag, scope)

        raw_files = glob(os.path.join(file_path, 'hdfra.*'))
        process_fn = lambda f: self._read_single_rawfile(f, galaxy_cls, logfile_path, time_lag, scope)
        results = process_files_in_parallel(raw_files, process_fn, max_workers)

        self.dataset = [r for r in results if r is not None]
        return self

    def read_hdf5(self, dir_path: str) -> "GalaxyDataSet":
        hdf5_files = glob(os.path.join(dir_path, '*.h5'))
        for file in tqdm(hdf5_files, desc="Reading HDF5 Files"):
            # for file in hdf5_files:
            frame = self._read_single_hdf5(file)
            if frame is not None:
                self.dataset.append(frame)
        return self

    def to_hdf5(self, dir_path: str) -> None:
        self._validate_nonempty()
        self._ensure_directory_exists(dir_path)

        for frame in self.dataset:
            save_path = os.path.join(dir_path, f"{extract_dump(frame.file_path):05d}.h5")
            frame.to_hdf5(save_path=save_path)

    def read_raw2hdf5(self, file_path: str, logfile_path: str, mode: str, time_lag: float, scope: float, dir_path: str,
                      max_workers: int = -1) -> None:
        galaxy_cls, time_lag, scope = self._initialize_para_set(mode, time_lag, scope)
        ensure_dir_exists(dir_path)
        raw_files = glob(os.path.join(file_path, 'hdfra.*'))

        process_fn = partial(self._read_single_raw2hdf5, galaxy_cls=galaxy_cls, logfile_path=logfile_path,
                             time_lag=time_lag, scope=scope, dir_path=dir_path)

        # process_fn = lambda f: self._read_single_raw2hdf5(f, galaxy_cls, logfile_path, time_lag, scope, dir_path)
        process_files_in_parallel(raw_files, process_fn, max_workers)

    def scale_transform(self, target_shape: Tuple[int, int]) -> "GalaxyDataSet":
        self._validate_nonempty()
        self._validate_target_shape(target_shape)

        transformed_dataset = GalaxyDataSet()
        for frame in tqdm(self.dataset, desc="Transforming Data to New Scale"):
            frame_transformed = self._scale_transform_frame(frame, target_shape)
            transformed_dataset.dataset.append(frame_transformed)

        return transformed_dataset

    def filter_by_mdot_macer(self, threshold: float) -> "GalaxyDataSet":
        self._validate_nonempty()
        filtered_data = GalaxyDataSet()
        for frame in self.dataset:
            if frame.mdot_macer is not None and frame.mdot_macer >= threshold:
                filtered_data.dataset.append(frame)
        return filtered_data

    def get_attr(self, attr_list: list[str] = None) -> np.ndarray:
        """
        Return an array of attributes for all frames.
        """
        self._validate_nonempty()
        attr_list = attr_list or self.dataset[0].galaxy_prop.attributes

        return np.array([[getattr(frame.galaxy_prop, attr) for attr in attr_list] for frame in self.dataset])

    @property
    def mdot_macer(self) -> np.ndarray:
        self._validate_nonempty()
        return np.array([frame.mdot_macer for frame in self.dataset])

    @property
    def mdot_bondi(self) -> np.ndarray:
        self._validate_nonempty()
        return np.array([frame.mdot_bondi for frame in self.dataset])

    @property
    def mbh(self) -> np.ndarray:
        self._validate_nonempty()
        return np.array([frame.mbh for frame in self.dataset])

    # ================= Internal Methods =================

    def _read_single_rawfile(self, file: str, galaxy_cls: Type[GalaxyData], logfile_path: str, time_lag: float,
                             scope: float) -> Union[GalaxyData, None]:
        try:
            dump = extract_dump(file)
            galaxy = galaxy_cls()
            galaxy.read_rawfile(file_path=os.path.dirname(file), dump=dump, logfile_path=logfile_path,
                                time_lag=time_lag, scope=scope)
            return galaxy
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return None

    def _read_single_raw2hdf5(self, file: str, galaxy_cls: Type[GalaxyData], logfile_path: str, time_lag: float,
                              scope: float, dir_path: str) -> None:
        try:
            dump = extract_dump(file)
            galaxy = galaxy_cls()
            galaxy.read_rawfile(file_path=os.path.dirname(file), dump=dump, logfile_path=logfile_path,
                                time_lag=time_lag, scope=scope)
            save_path = os.path.join(dir_path, f"{dump:05d}.h5")
            galaxy.to_hdf5(save_path=save_path)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return None

    @staticmethod
    def _read_single_hdf5(file) -> GalaxyData | None:
        try:
            galaxy = GalaxyData().read_hdf5(file_path=file)
            return galaxy
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            return None

    def _validate_nonempty(self) -> None:
        if not self.dataset:
            raise ValueError("The dataset is empty. Load data first.")

    @staticmethod
    def _initialize_para_set(mode: str, time_lag: float, scope: float):
        if mode == "eg":
            return EllipticalGalaxy, time_lag or 1e-4, scope or 5e-5
        elif mode == "dg":
            return DiskGalaxy, time_lag or 0.0125, scope or 5e-4
        else:
            raise ValueError("Unsupported mode. Must be 'eg' or 'dg'.")

    def _validate_target_shape(self, target_shape: tuple) -> None:
        if not (isinstance(target_shape, tuple) and len(target_shape) == 2 and
                all(isinstance(dim, int) and dim > 0 for dim in target_shape)):
            raise ValueError("Target shape must be a tuple of two positive integers.")

    @staticmethod
    def _scale_transform_frame(frame: GalaxyData, target_shape: Tuple[int, int]) -> GalaxyData:
        r_bins, theta_bins = target_shape
        r_len, theta_len = frame.shape
        if r_len % r_bins != 0 or theta_len % theta_bins != 0:
            raise ValueError("Target shape must evenly divide the original shape.")

        r_step, theta_step = r_len // r_bins, theta_len // theta_bins
        new_frame = GalaxyData()
        # Copy scalar attributes
        for attr in ["time", "mbh", "mdot_macer", "mdot_edd", "file_path", "logfile_path"]:
            setattr(new_frame, attr, getattr(frame, attr))

        # Downsampling spatial dimensions
        new_frame.r_vector = frame.r_vector[::r_step]
        new_frame.theta_vector = frame.theta_vector[::theta_step]

        # Weighted operations
        weights = frame.galaxy_prop.mass.reshape(r_bins, r_step, theta_bins, theta_step)

        for attr in frame.galaxy_prop.attributes:
            data = getattr(frame.galaxy_prop, attr)
            if attr in ["r", "theta", "volume", "mass", "density"]:
                # Special handling for these as sums or direct recalculation
                if attr in ["volume", "mass"]:
                    # sum over subdivided blocks
                    reshaped = data.reshape(r_bins, r_step, theta_bins, theta_step)
                    new_data = reshaped.sum(axis=(1, 3))
                elif attr == "density":
                    # density = mass / volume after aggregation
                    continue
                else:
                    # r, theta just take the coarser grid directly
                    new_data = data[::r_step, ::theta_step]
                setattr(new_frame.galaxy_prop, attr, new_data)
            else:
                # Weighted average for other attributes
                reshaped = data.reshape(r_bins, r_step, theta_bins, theta_step)
                new_data = np.average(reshaped, axis=(1, 3), weights=weights)
                setattr(new_frame.galaxy_prop, attr, new_data)

        # Recalculate volume, mass and density
        new_frame.galaxy_prop.density = new_frame.galaxy_prop.mass / new_frame.galaxy_prop.volume

        # Recalculate Bondi accretion
        new_frame._calculate_mdot_bondi()

        return new_frame

    def process_and_save(self, para: dict, target_shape=(8, 8), max_workers=-1):
        # Load and save raw data
        self.read_raw2hdf5(file_path=para['file_path'], logfile_path=para['logfile_path'], mode=para['mode'],
                           time_lag=para['time_lag'], scope=para['scope'], dir_path=para['fine_dir'],
                           max_workers=max_workers)

        # Transform and save coarse data
        GalaxyDataSet().read_hdf5(para['fine_dir']).scale_transform(target_shape).to_hdf5(para['coarse_dir'])

        # Filter and save filtered data
        GalaxyDataSet().read_hdf5(para['coarse_dir']).filter_by_mdot_macer(1e-5).to_hdf5(para['filtered_dir'])


def construct_file_paths(root_path, galaxy_type):
    """
    Construct all necessary file paths for processing a galaxy dataset.
    """
    return {
        'fine_dir': os.path.join(root_path, 'dataset', 'fine', galaxy_type),
        'coarse_dir': os.path.join(root_path, 'dataset', 'coarse', galaxy_type),
        'filtered_dir': os.path.join(root_path, 'dataset', 'filtered', galaxy_type),
    }


if __name__ == "__main__":
    load_dotenv()
    DATABASE_PATH = os.getenv("DATABASE_PATH")
    ROOT_PATH = os.getenv("ROOT_PATH")

    eg_para = {
        'file_path': os.path.join(DATABASE_PATH, 'elliptical_galaxy/elliptical_galaxy_21999/data'),
        'logfile_path': os.path.join(DATABASE_PATH, 'elliptical_galaxy/elliptical_galaxy_21999/zmp_usr'),
        'time_lag': 0.0001,
        'scope': 5e-5,
        'mode': "eg",
    }
    dg_para = {
        'file_path': os.path.join(DATABASE_PATH, 'disk_galaxy/fiducial/data/'),
        'logfile_path': os.path.join(DATABASE_PATH, 'disk_galaxy/fiducial/fiducial.log'),
        'time_lag': 0.0125,
        'scope': 5e-4,
        'mode': "dg",
    }
    dg_10_para = {
        'file_path': os.path.join(DATABASE_PATH, 'disk_galaxy/fiducial_10/data/'),
        'logfile_path': os.path.join(DATABASE_PATH, 'disk_galaxy/fiducial_10/fiducial_10.log'),
        'time_lag': 0.0125,
        'scope': 5e-4,
        'mode': "dg",
    }
    eg_para.update(construct_file_paths(ROOT_PATH, 'eg'))
    dg_para.update(construct_file_paths(ROOT_PATH, 'dg'))
    dg_10_para.update(construct_file_paths(ROOT_PATH, 'dg_10'))

    for para in [eg_para, dg_para, dg_10_para]:
        GalaxyDataSet().process_and_save(para, target_shape=(8, 8), max_workers=-1)

    for para in [eg_para, dg_para, dg_10_para]:
        GalaxyDataSet().read_hdf5(para['corase_dir']).filter_by_mdot_macer(1e-5).to_hdf5(para['filtered_dir'])
