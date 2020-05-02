import os
import sys
import numpy as np
import sqlite3
import pandas as pd
# Lenstronomy modules
#sys.path.insert(0, '/home/jwp/stage/sl/lenstronomy')
import lenstronomy
print("Lenstronomy path being used: {:s}".format(lenstronomy.__path__[0]))
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.param_util as param_util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from baobab.sim_utils import instantiate_PSF_models, get_PSF_model
from scipy.interpolate import interp1d

def to_csv(truth_db_path, dest_dir='.'):
    """Dumps sqlite3 files of the truth tables as csv files

    Parameters
    ----------
    truth_db_path : str
        Path to the truth tables Bryce made, either of the lens or the host

    """
    db = sqlite3.connect(truth_db_path)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(os.path.join(dest_dir, table_name + '.csv'), index=False)
    cursor.close()
    db.close()

def get_lambda_factor(ellip):
    """Get the interpolated lambda factor for the given Einstein radius that accounts for the ellipticity of projected mass

    Note
    ----
    The input data `ell_lef.dat` and this function are the work of Nan Li, based on Chae 2003

    Parameters
    ----------
    ellip : float
        the axis ratio defined as one minus minor/major axis

    Returns
    -------
    float
        the lambda factor with which to scale theta_E

    """
    e_tmp, lef_tmp = np.loadtxt("../../data/ell_lef.dat", comments='#', usecols=(0,1), unpack=True)
    interpolated_lambdas = interp1d(e_tmp, lef_tmp, kind='linear')
    return interpolated_lambdas(ellip)

def get_null_psf(pixel_scale):
    """Get a null (delta function) PSF

    Parameters
    ----------
    pixel_scale : float
        arcsec per pixel

    """
    psf_models = instantiate_PSF_models({'type': 'NONE'}, pixel_scale)
    psf_model = get_PSF_model(psf_models, len(psf_models), current_idx=0)
    return psf_model

def get_data_api(pixel_scale, num_pix):
    """Instantiate a simulation tool that knows the detector and observation conditions

    Parameters
    ----------
    pixel_scale : float
        arcsec per pixel
    num_pix : int
        number of pixels per side

    Returns
    -------
    Lenstronomy.DataAPI object

    """
    kwargs_detector = {'pixel_scale': pixel_scale, 'ccd_gain': 100.0, 'magnitude_zero_point': 30.0, 'exposure_time': 10000.0, 'psf_type': 'NONE', 'background_noise': 0.0}
    data_api = DataAPI(num_pix, **kwargs_detector)
    return data_api

def get_lens_params(lens_info, z_src, cosmo):
    """Get SIE lens parameters into a form Lenstronomy understands

    Parameters
    ----------
    lens_info : dict
        SIE lens and external shear parameters for a system, where `ellip_lens` is 1 minus the axis ratio 
    z_src : float
        source redshift, required for Einstein radius approximation
    cosmo : astropy.cosmology object
        cosmology to use to get distances

    """
    # Instantiate cosmology-aware models
    lens_cosmo = LensCosmo(z_lens=lens_info['redshift'], z_source=z_src, cosmo=cosmo)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(lens_info['vel_disp_lenscat'])
    lam = get_lambda_factor(lens_info['ellip_lens'])
    sie_mass = dict(
                      center_x=0.0,
                      center_y=0.0,
                      #s_scale=0.0,
                      theta_E=theta_E*lam,
                      e1=lens_info['e1_lens'],
                      e2=lens_info['e2_lens']
                      )
    return sie_mass

def get_external_shear_params(lens_info):
    """Get external shear parameters into a form Lenstronomy understands

    Parameters
    ----------
    lens_info : dict
        SIE lens and external shear parameters for a system, where `ellip_lenscat` is 1 minus the axis ratio 

    """
    external_shear = dict(
                          gamma_ext=lens_info['gamma_lenscat'],
                          psi_ext=lens_info['phig_rad_lenscat']
                          )
    #external_shear = dict(
    #                      gamma1=lens_info['shear_1_lenscat'],
    #                      gamma2=lens_info['shear_2_lenscat']
    #                      )
    return external_shear

def get_src_light_params(src_light_info, bulge_or_disk='bulge'):
    """Get Sersic parameters into a form Lenstronomy understands

    Parameters
    ----------
    src_light_info : dict
        Sersic host galaxy component (bulge or disk) parameters for a system

    """
    n_sersic = src_light_info['sindex_{:s}'.format(bulge_or_disk)]
    R_sersic = (src_light_info['major_axis_{:s}'.format(bulge_or_disk)]*src_light_info['minor_axis_{:s}'.format(bulge_or_disk)])**0.5
    q_src = src_light_info['minor_axis_{:s}'.format(bulge_or_disk)]/src_light_info['major_axis_{:s}'.format(bulge_or_disk)]
    e1_src, e2_src = param_util.phi_q2_ellipticity(src_light_info['position_angle_rad'], q_src)

    sersic_host = dict(
                     amp=20.0, # doesn't matter, image gets rescaled anyway
                     n_sersic=n_sersic,
                     R_sersic=R_sersic,
                     center_x=src_light_info['x_src'],
                     center_y=src_light_info['y_src'],
                     e1=e1_src,
                     e2=e2_src,
                     Re=0.1*R_sersic, # Core radius
                     gamma=0.0, # How much the profile flattens at the core
                     max_R_frac=5.0, # Zero mass outside of this region
                     alpha=99.0, # How abruptly the profile flattens at the core
                     )
    return sersic_host