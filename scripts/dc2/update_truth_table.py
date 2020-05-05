# -*- coding: utf-8 -*-
"""Updating the truth table
This script updates the lensing observables such as time delays and magnifications in truth tables of lensed AGN/SNe. The original truth tables are derived from existing catalogs of lensed objects but the lenses have been replaced with the DC2 galaxies so the observables are no longer consistent.

Example
-------
To run this script, pass in the path to the folder containing the `lens_truth.db` and `host_truth.db` files as the argument::
    
    $ python update_truth_table.py --input_dir example_truth_042420

The updated truth tables will be generated in the same folder.

"""
import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
# Lenstronomy modules
from astropy.cosmology import WMAP7, wCDM
#sys.path.insert(0, '/home/jwp/stage/sl/lenstronomy')
import lenstronomy
import lenstronomy.Util.param_util as param_util
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from baobab.sim_utils import generate_image_simple
import verification_utils as utils

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='directory of the original truth tables (and the OM10 catalog)', default='./example_truth_050120')
    parser.add_argument('--export_caustics', help='whether to export the caustics for lenses with discrepant number of images from OM10', default=False, type=bool)
    args = parser.parse_args()
    return args

def export_db(dataframe, out_dir, out_fname, table_name, overwrite=False):
    """Export a DB from a Pandas DataFrame

    Parameters
    ----------
    dataframe : Pandas.DataFrame object
    out_fname : str
    table_name : str
    overwrite_existing : bool
    
    """
    out_path = os.path.join(out_dir, out_fname)
    if overwrite is True and os.path.exists(out_path):
        os.remove(out_path)
    engine = create_engine('sqlite:///{:s}'.format(out_path), echo=False)
    dataframe.to_sql(table_name, con=engine, index=False)
    return None

def main():
    args = parse_args()
    input_dir = args.input_dir
    # Convert DB files into csv
    utils.to_csv(os.path.join(input_dir, 'lens_truth.db'), input_dir)
    utils.to_csv(os.path.join(input_dir, 'lensed_agn_truth.db'), input_dir)
    utils.to_csv(os.path.join(input_dir, 'host_truth.db'), input_dir)
    # Convert to dataframes for easy manipulation
    lens_df = pd.read_csv(os.path.join(input_dir, 'agn_lens.csv'), index_col=None) # SIE lens mass
    agn_df = pd.read_csv(os.path.join(input_dir, 'lensed_agn.csv'), index_col=None) # AGN light 
    src_light_df = pd.read_csv(os.path.join(input_dir, 'agn_hosts.csv'), index_col=None) # Host galaxy light

    #####################
    # Model assumptions #
    #####################
    # Cosmology
    #cosmo = WMAP7 # DC2
    cosmo = wCDM(H0=72.0, Om0=0.26, Ode0=0.74, w0=-1.0) # OM10
    # Imaging
    pixel_scale = 0.01 # arcsec/pix, hardcoded postage stamp config by Nan and Matt
    num_pix = 1000
    null_psf = utils.get_null_psf(pixel_scale) # delta function PSF
    data_api = utils.get_data_api(pixel_scale, num_pix) # simulation tool for generating images
    # Density profiles
    kwargs_model = {}
    kwargs_model['lens_model_list'] = ['SIE', 'SHEAR_GAMMA_PSI']
    kwargs_model['point_source_model_list'] = ['LENSED_POSITION']
    src_light_model = LightModel(['CORE_SERSIC'])
    
    #########################
    # Basic unit conversion #
    # OM10 --> Lenstronomy  #
    #########################
    arcsec_to_deg = 1/3600.0
    # SIE lens mass
    lens_phie_rad = np.pi*(lens_df['phie_lens']/180.0) + 0.5*np.pi # in rad, origin at y axis
    lens_e1, lens_e2 = param_util.phi_q2_ellipticity(lens_phie_rad, 1 - lens_df['ellip_lens'])
    lens_df['e1_lens'] = lens_e1
    lens_df['e2_lens'] = lens_e2
    lens_df['phie_rad_lens'] = lens_phie_rad
    # External shear
    lens_df['phig_rad_lenscat'] = np.deg2rad(lens_df['phig_lenscat'])
    # Sersic host light
    src_light_df['position_angle_rad'] = 0.5*np.deg2rad(src_light_df['position_angle'])

    # Temporarily read in OM10 for testing (to restore OM10 lens redshift)
    om10_path = os.path.join(input_dir, 'om10_qso_mock.fits')
    om10 = fits.open(om10_path)[1].data
    col_names = ['LENSID', 'ELLIP', 'PHIE', 'GAMMA', 'PHIG', 'ZLENS', 'ZSRC', 'VELDISP', 'XSRC', 'YSRC', 'NIMG',]
    df_data = {}
    for col in col_names:
        df_data[col] = om10[col].byteswap().newbyteorder()
    df_data['x_image'] = om10['XIMG'].tolist()
    df_data['y_image'] = om10['YIMG'].tolist()
    df_data['time_delays'] = om10['DELAY'].tolist()
    om10_df = pd.DataFrame(df_data)

    discrepancies = []
    sys_ids = lens_df['lens_cat_sys_id'].unique()
    progress = tqdm(total=len(sys_ids))
    for i, sys_id in enumerate(sys_ids):
        # Slice relevant params
        lens_info = lens_df[lens_df['lens_cat_sys_id']==sys_id].squeeze()
        agn_info = agn_df[agn_df['lens_cat_sys_id'] == sys_id].iloc[[0]].copy() # df of length 1
        agn_info_full = agn_df[agn_df['lens_cat_sys_id'] == sys_id].copy() # of length 2 or 4
        agn_df_index = agn_info_full[agn_info_full['lens_cat_sys_id'] == sys_id].index # length 2 or 4
        agn_df.drop(agn_df_index, axis=0, inplace=True) # delete for now, will add back
        src_light_info = src_light_df.loc[src_light_df['lens_cat_sys_id']==sys_id].iloc[0].squeeze() # arbitarily take the first lensed image, since the source properties are the same across the images
        om10_info = om10_df[om10_df['LENSID'] == sys_id].T.squeeze()
        z_lens = om10_info['ZLENS']#lens_info['redshift']
        z_src = src_light_info['redshift']
        x_lens = lens_info['ra_lens'] # absolute position in rad
        y_lens = lens_info['dec_lens']
        x_src = src_light_info['x_src'] # defined relative to lens center
        y_src = src_light_info['y_src']

        #######################
        # Solve lens equation #
        #######################
        lens_mass_model = LensModel(['SIE', 'SHEAR_GAMMA_PSI',], cosmo=cosmo, z_lens=z_lens, z_source=z_src)
        lens_eq_solver = LensEquationSolver(lens_mass_model)
        SIE = utils.get_lens_params(lens_info, z_src=z_src, cosmo=cosmo)
        external_shear = utils.get_external_shear_params(lens_info)
        kwargs_lens_mass = [SIE, external_shear]
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
        if n_img != agn_info_full.shape[0]:
            discrepancies.append(sys_id)
            if args.export_caustics:
                # Image the bulge
                all_kwargs = {}
                all_kwargs['lens_mass'] = utils.get_lens_params(lens_info, z_src=z_src, cosmo=cosmo)
                all_kwargs['external_shear'] = utils.get_external_shear_params(lens_info)
                all_kwargs['src_light'] = utils.get_src_light_params(src_light_info, bulge_or_disk='bulge')
                bulge_img, _ = generate_image_simple(all_kwargs, null_psf, data_api, lens_mass_model, src_light_model, lens_eq_solver, pixel_scale, num_pix, ['lens_mass', 'external_shear', 'src_light',],  {'supersampling_factor': 1}, min_magnification=0.0, lens_light_model=None, ps_model=None,)
                bulge_img /= np.max(bulge_img)
                # Overlay critical curves and caustics
                fig, ax = plt.subplots()
                crop_pix = 350
                eff_num_pix = num_pix - 2*crop_pix
                ax = utils.lens_model_plot_custom(bulge_img[crop_pix:-crop_pix,crop_pix:-crop_pix], ax, lensModel=lens_mass_model, kwargs_lens=kwargs_lens_mass, sourcePos_x=x_src, sourcePos_y=y_src, point_source=True, with_caustics=True, deltaPix=pixel_scale, numPix=eff_num_pix)
                # Overlay OM10 image positions
                n_image = om10_info['NIMG']
                om10_x_image = np.array(om10_info['x_image'][:n_image])
                om10_y_image = np.array(om10_info['y_image'][:n_image])
                ax.scatter(om10_x_image + eff_num_pix*pixel_scale*0.5, om10_y_image + eff_num_pix*pixel_scale*0.5, color='white', marker='*', alpha=0.5)
                ax.axis('off')
                fig.savefig('{:d}.png'.format(sys_id))
                plt.close()
            
        #################################
        # Update lensed AGN truth table #
        #################################
        agn_info = agn_info.loc[agn_info.index.repeat(n_img)] # replicate enough rows
        # Absolute image positions in rad
        ra_image_abs = x_lens + np.radians(x_image*arcsec_to_deg)
        dec_image_abs = y_lens + np.radians(y_image*arcsec_to_deg) # FIXME: use the validated LSST function? cos(dec) factor not included for now
        # Reorder the existing images by increasing dec to ensure consistency between OM10 and DC2
        # Only 'unique_id', 'image_number', ra', 'dec', 't_delay', 'magnification' are affected by the reordering
        increasing_dec_i = np.argsort(dec_image_abs)
        agn_info['unique_id'] = ['{:s}_{:d}'.format(agn_info.iloc[0]['dc2_sys_id'], img_i) for img_i in range(n_img)] 
        agn_info['image_number'] = np.arange(n_img)
        agn_info['ra'] = ra_image_abs[increasing_dec_i]
        agn_info['dec'] = dec_image_abs[increasing_dec_i]
        agn_info['magnification'] = magnification[increasing_dec_i]
        time_delays = time_delays[increasing_dec_i]
        time_delays -= time_delays[0] # time delays relative to first image
        agn_info['t_delay'] = time_delays

        #agn_df.update(agn_info) # inplace op doesn't work when n_img is different
        agn_df = agn_df.append(agn_info, ignore_index=True, sort=False)
        agn_df.reset_index(drop=True, inplace=True) # to prevent duplicate indices
        progress.update(1)
    # Sort by dc2_sys_id and image number
    agn_df['dc2_sys_id_int'] = [int(sys_id.split('_')[-1]) for sys_id in agn_df['dc2_sys_id']]
    agn_df.sort_values(['dc2_sys_id_int', 'image_number'], axis=0, inplace=True)
    agn_df.drop(['dc2_sys_id_int'], axis=1, inplace=True)
    # Export to original file format
    agn_df.to_csv('new_agn_df.csv', index=False)
    export_db(agn_df, input_dir, 'updated_lensed_agn_truth.db', 'lensed_agn', overwrite=True)
    progress.close()

    with open(os.path.join("image_number_discrepancies.txt"), "w") as f:
        for lens_i in discrepancies:
            f.write(str(lens_i) +"\n")

if __name__ == '__main__':
    main()
