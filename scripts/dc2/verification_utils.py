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
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants
from astropy.cosmology import FlatLambdaCDM
from  lenstronomy.Plots import lens_plot
import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Plots import plot_util
import scipy.ndimage as ndimage
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


#TODO define coordinate grid beforehand, e.g. kwargs_data
def lens_model_plot_custom(image, ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0, point_source=False, with_caustics=False):
    """
    Overlay the critical curves and caustics over the lensed image

    :param ax:
    :param kwargs_lens:
    :param numPix:
    :param deltaPix:
    :return:
    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix
    x_grid, y_grid = data.pixel_coordinates
    lensModelExt = LensModelExtensions(lensModel)
    #ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExt.critical_curve_caustics(
    #    kwargs_lens, compute_window=_frame_size, grid_scale=deltaPix/2.)
    #x_grid1d = util.image2array(x_grid)
    #y_grid1d = util.image2array(y_grid)
    #kappa_result = lensModel.kappa(x_grid1d, y_grid1d, kwargs_lens)
    #kappa_result = util.array2image(kappa_result)
    #im = ax.matshow(np.log10(kappa_result), origin='lower', extent=[0, _frame_size, 0, _frame_size], cmap='Greys',vmin=-1, vmax=1) #, cmap=self._cmap, vmin=v_min, vmax=v_max)
    _ = ax.matshow(image, origin='lower', extent=[0, _frame_size, 0, _frame_size])
    if with_caustics is True:
        ra_crit_list, dec_crit_list = lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=_frame_size, start_scale=deltaPix, max_order=20)
        ra_caustic_list, dec_caustic_list = lensModel.ray_shooting(ra_crit_list, dec_crit_list, kwargs_lens)
        plot_util.plot_line_set(ax, _coords, ra_caustic_list, dec_caustic_list, color='y')
        plot_util.plot_line_set(ax, _coords, ra_crit_list, dec_crit_list, color='r')
    if point_source:
        solver = LensEquationSolver(lensModel)
        theta_x, theta_y = solver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                             min_distance=deltaPix, search_window=deltaPix*numPix)
        mag_images = lensModel.magnification(theta_x, theta_y, kwargs_lens)
        x_image, y_image = _coords.map_coord2pix(theta_x, theta_y)
        abc_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        for i in range(len(x_image)):
            x_ = (x_image[i] + 0.5) * deltaPix
            y_ = (y_image[i] + 0.5) * deltaPix
            ax.plot(x_, y_, 'dk', markersize=4*(1 + np.log(np.abs(mag_images[i]))), alpha=0.5)
            ax.text(x_, y_, abc_list[i], fontsize=20, color='k')
        x_source, y_source = _coords.map_coord2pix(sourcePos_x, sourcePos_y)
        ax.plot((x_source + 0.5) * deltaPix, (y_source + 0.5) * deltaPix, '*r', markersize=5)
    #ax.plot(numPix * deltaPix*0.5 + pred['lens_mass_center_x'] + pred['src_light_center_x'], numPix * deltaPix*0.5 + pred['lens_mass_center_y'] + pred['src_light_center_y'], '*k', markersize=5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    return ax

def to_csv(truth_db_path, dest_dir='.', table_suffix=''):
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
        table.to_csv(os.path.join(dest_dir, table_name + '{:s}.csv'.format(table_suffix)), index=False)
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
    e_tmp, lef_tmp = np.loadtxt('/home/jwp/stage/sl/LatestSLSprinkler/data/ell_lef.dat', comments='#', usecols=(0,1), unpack=True)
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
    # I removed lam because lenstronomy accepts the spherically-averaged Einstein radius as input.
    phi, q = param_util.ellipticity2phi_q(lens_info['e1_lens'], lens_info['e2_lens'])
    # Factor converting the grav lens ellipticity convention (square average) to lenstronomy's (product average)
    gravlens_to_lenstronomy = np.sqrt((1.0 + q**2.0)/(2.0*q))
    sie_mass = dict(
                      center_x=0.0,
                      center_y=0.0,
                      #s_scale=0.0,
                      theta_E=theta_E*gravlens_to_lenstronomy,
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