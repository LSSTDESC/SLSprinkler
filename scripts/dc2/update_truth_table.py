# -*- coding: utf-8 -*-
"""Updating the truth table
This script updates the lensing observables such as time delays and magnifications in truth tables of lensed AGN/SNe. The original truth tables are derived from existing catalogs of lensed objects but the lenses have been replaced with the DC2 galaxies so the observables are no longer consistent.

Example
-------
To use the default settings, run this script from the root of the repo with the required `object_type` argument::

    $ python update_truth_table.py agn

The fits files will be generated in the original `datadir`.

"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine
from astropy.cosmology import WMAP7, wCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lensing_utils
import io_utils

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser(description="Script to generate postage-stamp fits files of lensed AGN hosts")
    parser.add_argument("object_type", type=str,
                        help="Type of object the source galaxy hosts ('agn' or 'sne')")
    parser.add_argument("--datadir", type=str, default='truth_tables',
                    help='Location of directory containing truth tables')
    parser.add_argument("--pixel_size", type=float, default=0.04,
                        help='Pixel size in arcseconds. Used to set the numerical precision of the lens equation solver.')
    parser.add_argument("--num_pix", type=int, default=250,
                        help='Number of pixels in x- or y-direction. Used to set the numerical precision of the lens equation solver.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_dir = args.datadir
    object_type = args.object_type
    # Load DB files as dataframes
    lens_df = pd.read_sql('%s_lens' % object_type, os.path.join('sqlite:///', input_dir, 'lens_truth.db'), index_col=0)
    ps_df = pd.read_sql('lensed_%s' % object_type,
                        os.path.join('sqlite:///', input_dir, 'lensed_%s_truth.db' % object_type), index_col=0)
    src_light_df = pd.read_sql('%s_hosts' % object_type,
                               os.path.join('sqlite:///', input_dir, 'host_truth.db'), index_col=0)

    #####################
    # Model assumptions #
    #####################
    # Cosmology
    cosmo = WMAP7 # DC2
    #cosmo = wCDM(H0=72.0, Om0=0.26, Ode0=0.74, w0=-1.0) # OM10
    # Imaging config, only used to set the numerics of the lens equation solver
    pixel_scale = args.pixel_size
    num_pix = args.num_pix
    # Density profiles
    kwargs_model = {}
    kwargs_model['lens_model_list'] = ['SIE', 'SHEAR_GAMMA_PSI']
    kwargs_model['point_source_model_list'] = ['LENSED_POSITION']
    arcsec_to_deg = 1/3600.0

    sys_ids = lens_df['lens_cat_sys_id'].unique()
    progress = tqdm(total=len(sys_ids))
    for i, sys_id in enumerate(sys_ids):
        # Slice relevant params
        lens_info = lens_df[lens_df['lens_cat_sys_id']==sys_id].squeeze()
        ps_info = ps_df[ps_df['lens_cat_sys_id'] == sys_id].iloc[[0]].copy() # df of length 1
        ps_info_full = ps_df[ps_df['lens_cat_sys_id'] == sys_id].copy() # of length 2 or 4
        ps_df_index = ps_info_full[ps_info_full['lens_cat_sys_id'] == sys_id].index # length 2 or 4
        ps_df.drop(ps_df_index, axis=0, inplace=True) # delete for now, will add back
        src_light_info = src_light_df.loc[src_light_df['lens_cat_sys_id']==sys_id].iloc[0].squeeze() # arbitarily take the first lensed image, since the source properties are the same across the images
        z_lens = lens_info['redshift']
        z_src = src_light_info['redshift']
        x_lens = lens_info['ra_lens'] # absolute position in rad
        y_lens = lens_info['dec_lens']
        # FIXME: further adjustment needed here for the SNe, as the SNe are offset from the host center
        x_src = ps_info['x_%s' % object_type].values[0] # defined relative to lens center
        y_src = ps_info['y_%s' % object_type].values[0]

        #######################
        # Solve lens equation #
        #######################
        lens_mass_model = LensModel(['SIE', 'SHEAR_GAMMA_PSI',], cosmo=cosmo, z_lens=z_lens, z_source=z_src)
        lens_eq_solver = LensEquationSolver(lens_mass_model)
        kwargs_lens_mass = lensing_utils.get_lens_params(lens_info, z_src=z_src, cosmo=cosmo)
        x_image, y_image = lens_eq_solver.findBrightImage(x_src, y_src,
                                                          kwargs_lens_mass,
                                                          min_distance=0.01, # default is 0.01
                                                          numImages=4,
                                                          search_window=num_pix*pixel_scale, # default is 5
                                                          precision_limit=10**(-10) # default
                                                          )
        magnification = np.abs(lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass))
        td_cosmo = TDCosmography(z_lens, z_src, kwargs_model, cosmo_fiducial=cosmo)
        ps_kwargs = [{'ra_image': x_image, 'dec_image': y_image}]
        time_delays = td_cosmo.time_delays(kwargs_lens_mass, ps_kwargs, kappa_ext=0.0)
        n_img = len(x_image)

        #################################
        # Update lensed AGN truth table #
        #################################
        ps_info = ps_info.loc[ps_info.index.repeat(n_img)] # replicate enough rows
        # Absolute image positions in rad
        ra_image_abs = x_lens + np.radians(x_image*arcsec_to_deg)
        dec_image_abs = y_lens + np.radians(y_image*arcsec_to_deg) # FIXME: use the validated LSST function? cos(dec) factor not included for now
        # Reorder the existing images by increasing dec to enforce a consistent image ordering system
        # Only 'unique_id', 'image_number', ra', 'dec', 't_delay', 'magnification' are affected by the reordering
        increasing_dec_i = np.argsort(dec_image_abs)
        ps_info['unique_id'] = ['{:s}_{:d}'.format(ps_info.iloc[0]['dc2_sys_id'], img_i) for img_i in range(n_img)]
        ps_info['image_number'] = np.arange(n_img)
        ps_info['ra'] = ra_image_abs[increasing_dec_i]
        ps_info['dec'] = dec_image_abs[increasing_dec_i]
        ps_info['magnification'] = magnification[increasing_dec_i]
        time_delays = time_delays[increasing_dec_i]
        time_delays -= time_delays[0] # time delays relative to first image
        ps_info['t_delay'] = time_delays
        #ps_df.update(ps_info) # inplace op doesn't work when n_img is different from OM10
        ps_df = ps_df.append(ps_info, ignore_index=True, sort=False)
        ps_df.reset_index(drop=True, inplace=True) # to prevent duplicate indices
        progress.update(1)
    # Sort by dc2_sys_id and image number
    ps_df['dc2_sys_id_int'] = [int(sys_id.split('_')[-1]) for sys_id in ps_df['dc2_sys_id']]
    ps_df.sort_values(['dc2_sys_id_int', 'image_number'], axis=0, inplace=True)
    ps_df.drop(['dc2_sys_id_int'], axis=1, inplace=True)
    # Export to original file format
    engine = create_engine(os.path.join('sqlite:///', input_dir, 'updated_lensed_%s_truth.db' % object_type), echo=False)
    ps_df.to_sql('lensed_%s' % object_type, con=engine)
    progress.close()

if __name__ == '__main__':
    main()
