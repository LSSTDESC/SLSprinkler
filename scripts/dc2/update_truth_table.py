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
import sys
sys.path.append('.')
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.cosmology import WMAP7, wCDM
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lensing_utils
import io_utils
from sprinkler import DC2Sprinkler

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
    lens_df = pd.read_sql(f'{object_type}_lens', str('sqlite:///' + os.path.join(input_dir, 'lens_truth.db')), index_col=0)
    ps_df = pd.read_sql(f'lensed_{object_type}', str('sqlite:///' + os.path.join(input_dir, f'lensed_{object_type}_truth.db')), index_col=0)
    src_light_df = pd.read_sql(f'{object_type}_hosts', str('sqlite:///' + os.path.join(input_dir, 'host_truth.db')), index_col=0)
    # Init columns to add
    ps_df['total_magnification'] = np.nan 
    src_light_df['total_magnification_bulge'] = np.nan 
    src_light_df['total_magnification_disk'] = np.nan
    for band in list('ugrizy'):
        src_light_df[f'lensed_flux_{band}'] = np.nan
        src_light_df[f'lensed_flux_{band}_noMW'] = np.nan
    dc2_sprinkler = DC2Sprinkler() # utility class for flux integration

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
    # Instantiate tool for imaging our hosts
    lensed_host_imager = lensing_utils.LensedHostImager(pixel_scale, num_pix)
    sys_ids = lens_df['lens_cat_sys_id'].unique()
    progress = tqdm(total=len(sys_ids))
    for i, sys_id in enumerate(sys_ids):
        #######################
        # Slice relevant rows #
        #######################
        # Lens mass
        lens_info = lens_df[lens_df['lens_cat_sys_id']==sys_id].iloc[0].squeeze()
        # Lensed point source
        ps_info = ps_df[ps_df['lens_cat_sys_id'] == sys_id].iloc[[0]].copy() # sub-df of length 1, to be updated
        ps_df.drop(ps_df[ps_df['lens_cat_sys_id'] == sys_id].index, axis=0, inplace=True) # delete rows for this system for now, will add back
        # Host light
        src_light_read_only = src_light_df.loc[src_light_df['lens_cat_sys_id']==sys_id].iloc[0].squeeze() # for accessing the original host info; arbitarily take the first lensed image, since properties we use are same across the images
        src_light_info = src_light_df[src_light_df['lens_cat_sys_id'] == sys_id].iloc[[0]].copy() # sub-df of length 1, to be updated
        src_light_df.drop(src_light_df[src_light_df['lens_cat_sys_id'] == sys_id].index, axis=0, inplace=True) # delete rows for this system for now, will add back
        # Properties defining lens geometry
        z_lens = lens_info['redshift']
        z_src = src_light_read_only['redshift']
        x_lens = lens_info['ra_lens'] # absolute position in rad
        y_lens = lens_info['dec_lens']
        x_src = ps_info[f'x_{object_type}'].values[0] # defined in arcsec wrt lens center
        y_src = ps_info[f'y_{object_type}'].values[0]
        x_src_host = src_light_read_only['x_src'] # defined in arcsec wrt lens center
        y_src_host = src_light_read_only['y_src']

        ########################################
        # Solve lens equation for point source #
        ########################################
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
        ps_info = ps_info.loc[ps_info.index.repeat(n_img)].reset_index(drop=True) # replicate enough rows
        # Absolute image positions in rad
        ra_image_abs = x_lens + x_image*arcsec_to_deg/np.cos(np.radians(y_lens))
        dec_image_abs = y_lens + y_image*arcsec_to_deg
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
        # FIXME: Check that the following works to update fluxes correctly
        if object_type == 'agn': # since AGN follow a single SED template
            agn_magnorm = ps_info['magnorm'].iloc[0] # unlensed magnorm, same across images
            for agn_idx, agn_magnorm in list(enumerate(ps_info['magnorm'].values)):
                dmag = -2.5*np.log10(np.abs(ps_info['magnification'].iloc[agn_idx]))
                magnorm_dict = {band: agn_magnorm + dmag for band in ['u', 'g', 'r', 'i', 'z', 'y']}
                agn_flux_no_mw, agn_flux_mw = dc2_sprinkler.add_flux('agnSED/agn.spec.gz',
                                                                    z_src,
                                                                    magnorm_dict, lens_info['av_mw'],
                                                                    lens_info['rv_mw'])
                for band in list('ugrizy'):
                    ps_info.loc[agn_idx, f'flux_{band}_agn'] = agn_flux_mw[band]
                    ps_info.loc[agn_idx, f'flux_{band}_agn_noMW'] = agn_flux_no_mw[band]

        ps_info['total_magnification'] = np.sum(np.abs(magnification))
        ps_df = ps_df.append(ps_info, ignore_index=True, sort=False)
        ps_df.reset_index(drop=True, inplace=True) # to prevent duplicate indices

        #########################################
        # Solve lens equation for host centroid #
        #########################################
         # Solve the lens equation for a hypothetical point source at the host centroid
        x_image_host, y_image_host = lens_eq_solver.findBrightImage(x_src_host, y_src_host,
                                                                  kwargs_lens_mass,
                                                                  min_distance=0.0075, # default is 0.01
                                                                  numImages=4,
                                                                  search_window=num_pix*pixel_scale, # default is 5
                                                                  precision_limit=10**(-10) # default
                                                                  )
        # Absolute image positions in rad
        ra_image_abs_host = x_lens + x_image_host*arcsec_to_deg/np.cos(np.radians(y_lens))
        dec_image_abs_host = y_lens + y_image_host*arcsec_to_deg
        n_img_host = len(y_image_host)
        
        ###########################
        # Update host truth table #
        ###########################
        src_light_info = src_light_info.loc[src_light_info.index.repeat(n_img_host)].reset_index(drop=True) # replicate enough rows
        # Reorder the existing images by increasing dec to enforce a consistent image ordering system
        increasing_dec_i_host = np.argsort(y_image_host)
        src_light_info['unique_id'] = ['{:s}_{:d}'.format(src_light_read_only['dc2_sys_id'], img_i) for img_i in range(n_img_host)]
        src_light_info['image_number'] = np.arange(n_img_host)
        src_light_info['ra_host_lensed'] = ra_image_abs_host[increasing_dec_i_host]
        src_light_info['dec_host_lensed'] = dec_image_abs_host[increasing_dec_i_host]
        src_light_info['x_img'] = x_image_host[increasing_dec_i_host]
        src_light_info['y_img'] = y_image_host[increasing_dec_i_host]

        bulge_img, bulge_features = lensed_host_imager.get_image(lens_info, src_light_read_only, z_lens, z_src, 'bulge')
        disk_img, disk_features = lensed_host_imager.get_image(lens_info, src_light_read_only, z_lens, z_src, 'disk')
        # Update magnorms based on lensed and unlensed flux
        for band in list('ugrizy'):
            src_light_info[f'magnorm_bulge_{band}'] = bulge_features['magnorms'][band]
            src_light_info[f'magnorm_disk_{band}'] = disk_features['magnorms'][band]
        src_light_info['total_magnification_bulge'] = bulge_features['total_magnification']
        src_light_info['total_magnification_disk'] = disk_features['total_magnification']

        disk_flux_no_mw, disk_flux_mw = dc2_sprinkler.add_flux(src_light_read_only['sed_disk_host'][2:-1],
                                                               z_src,
                                                               disk_features['magnorms'], lens_info['av_mw'],
                                                               lens_info['rv_mw'])

        bulge_flux_no_mw, bulge_flux_mw = dc2_sprinkler.add_flux(src_light_read_only['sed_bulge_host'][2:-1],
                                                               z_src,
                                                               bulge_features['magnorms'], lens_info['av_mw'],
                                                               lens_info['rv_mw'])
        for band in list('ugrizy'):
            src_light_info[f'lensed_flux_{band}'] = disk_flux_mw[band] + bulge_flux_mw[band]
            src_light_info[f'lensed_flux_{band}_noMW'] = disk_flux_no_mw[band] + bulge_flux_no_mw[band]

        src_light_df = src_light_df.append(src_light_info, ignore_index=True, sort=False)
        src_light_df.reset_index(drop=True, inplace=True) # to prevent duplicate indices

        progress.update(1)
    # Sort by dc2_sys_id and image number
    ps_df['dc2_sys_id_int'] = [int(sys_id.split('_')[-1]) for sys_id in ps_df['dc2_sys_id']]
    ps_df.sort_values(['dc2_sys_id_int', 'image_number'], axis=0, inplace=True)
    ps_df.drop(['dc2_sys_id_int'], axis=1, inplace=True)
    src_light_df['dc2_sys_id_int'] = [int(sys_id.split('_')[-1]) for sys_id in src_light_df['dc2_sys_id']]
    src_light_df.sort_values(['dc2_sys_id_int', 'image_number'], axis=0, inplace=True)
    src_light_df.drop(['dc2_sys_id_int'], axis=1, inplace=True)
    # Export lensed_ps and host truth tables to original file format
    io_utils.export_db(ps_df, input_dir, f'updated_lensed_{object_type}_truth.db', f'lensed_{object_type}', overwrite=True)
    io_utils.export_db(src_light_df, input_dir, f'updated_host_truth.db', f'{object_type}_hosts', overwrite=True)
    progress.close()

if __name__ == '__main__':
    main()
