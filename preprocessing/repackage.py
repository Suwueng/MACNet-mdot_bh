import os
import numpy as np

import toolbox as tb
from toolbox.dataload import EllipticalGalaxyMdot, DiskGalaxyMdot, EllipticalGalaxyData

if __name__ == '__main__':
    dump_list_eg = list(range(1, 22000))
    dump_list_dg = list(range(46, 1000))
    suffix = 'h5'

    # Create Fine Dataset
    tb.createdataset.eg_fine_dataset(
        folder_path=os.path.join(tb.database_path, 'elliptical_galaxy/elliptical_galaxy_21999'),
        dumps_list=dump_list_eg,
        save_path=os.path.join(tb.root_path, 'dataset/fine/eg_21999'),
        suffix=suffix,
        m_bh=4.5e9 / 2.5e7,
        time_lag=0.0001
    )
    tb.createdataset.dg_fine_dataset(
        folder_path=os.path.join(tb.database_path, 'disk_galaxy/fiducial'),
        dumps_list=dump_list_dg,
        save_path=os.path.join(tb.root_path, 'dataset/fine/dg_954'),
        suffix=suffix,
        m_bh=5e7 / 2.5e7,
        time_lag=0.0125
    )

    # Create Coarse Dataset
    tb.createdataset.coarse_dataset(
        folder_path=os.path.join(tb.root_path, 'dataset/fine/eg_21999'),
        dumps_list=dump_list_eg,
        theta_scale=8,
        r_scale=32,
        save_path=os.path.join(tb.root_path, 'dataset/coarse/eg_21999'),
        suffix=suffix
    )
    tb.createdataset.coarse_dataset(
        folder_path=os.path.join(tb.root_path, 'dataset/fine/dg_954'),
        dumps_list=dump_list_dg,
        theta_scale=9,
        r_scale=35,
        save_path=os.path.join(tb.root_path, 'dataset/coarse/dg_954'),
        suffix=suffix
    )

    # Create Black hole accretion rate dataset
    eg_mdot = EllipticalGalaxyMdot(time_lag=0.0001) \
        .load(os.path.join(tb.database_path, 'elliptical_galaxy/elliptical_galaxy_21999/zmp_usr')) \
        .process(dumps_list=dump_list_eg, scope=5e-5) \
        .to_csv(os.path.join(tb.root_path, 'dataset/mdot_bh_eg_21999_5e-5.csv'), index=False)
    dg_mdot = DiskGalaxyMdot(time_lag=0.0125) \
        .load(os.path.join(tb.database_path, 'disk_galaxy/fiducial.log')) \
        .process(dumps_list=dump_list_dg, scope=5e-4) \
        .to_csv(os.path.join(tb.root_path, 'dataset/mdot_bh_dg_954_5e-4.csv'), index=False)

    # Encapsulate data as .npy file
        # Elliptical galaxy
    path = os.path.join(tb.root_path, 'dataset/coarse/eg_21999')
    x, y, bondi = tb.createdataset.encapsulate_data(
        folder_path=path,
        frame_num=1,
        suffix=suffix,
        y_pattern='mdot_bh_eg_21999_5e-5.csv',
        r_acc=[0.3, 0.03],
        gamma=5/3,
        gamma1=5/3 - 1,
        w='m'
    )
    x = np.squeeze(x, axis=1)
    np.save(os.path.join(path, 'x_1.npy'), x)
    np.save(os.path.join(path, 'y_1.npy'), y)
    np.save(os.path.join(path, 'bondi_accretion_rate_0.3_0.03.npy'), bondi)

        # Disk galaxy
    path = os.path.join(tb.root_path, 'dataset/coarse/dg_954')
    x, y, bondi = tb.createdataset.encapsulate_data(
        folder_path=path,
        frame_num=1,
        y_pattern='mdot_bh_dg_954_5e-4.csv',
        r_acc=[0.3],
        gamma=5/3,
        gamma1=5/3 - 1,
        w='m'
    )
    x = np.squeeze(x, axis=1)
    np.save(os.path.join(path, 'x_1.npy'), x)
    np.save(os.path.join(path, 'y_1.npy'), y)
    np.save(os.path.join(path, 'bondi_accretion_rate_0.3.npy'), bondi)

    # =================================================================================================================
    # # Data visualization
    # eg3000 = EllipticalGalaxyData(os.path.join(tb.database_path, 'elliptical_galaxy_3379'), 3000)
    # tb.dataplot.plot_contourf(eg3000.r, eg3000.theta, eg3000.galaxy_prop.abun_wind, label='abun_wind')
    #
    # # Resolution conversion diagram
    # eg3000 = tb.dataload.read_pickle(os.path.join(tb.root_path, 'dataset/fine/eg_3379/03000.pkl'))
    # tb.dataplot.demo_fine2coarse(eg3000, prop='dnewstar',
    #                              filter_sizes=[(1, 2), (1, 4), (2, 8), (4, 16), (8, 32)])
