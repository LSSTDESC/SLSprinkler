#!/usr/bin/env python
"""Utility functions related to lensing parameter conversions and imaging

"""
import numpy as np
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.param_util as param_util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from scipy.interpolate import interp1d
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import WMAP7, wCDM
from lenstronomy.Data.psf import PSF

class LensedHostImager:
    """Image generator for lensed hosts, bulge or disk

    Parameters
    ----------
    pixel_scale : float
    num_pix : int

    """
    def __init__(self, pixel_scale, num_pix):
        self.null_psf = get_null_psf(pixel_scale) # delta function PSF
        self.data_api = get_data_api(pixel_scale, num_pix) # simulation tool for generating images
        self.cosmo = WMAP7 # DC2
        #self.cosmo = wCDM(H0=72.0, Om0=0.26, Ode0=0.74, w0=-1.0) # OM10
        self.src_light_model = LightModel(['SERSIC_ELLIPSE'])
        self.cored_sersic_model = LightModel(['CORE_SERSIC'])
        self.bands = list('ugrizy')

    def get_image(self, lens_info, src_light_info, z_lens, z_src, bulge_or_disk):
        lens_mass_model = LensModel(['SIE', 'SHEAR_GAMMA_PSI',], cosmo=self.cosmo, z_lens=z_lens, z_source=z_src)
        lens_mass_kwargs = get_lens_params(lens_info, z_src=z_src, cosmo=self.cosmo)
        src_light_kwargs = get_src_light_params(src_light_info, bulge_or_disk=bulge_or_disk) 
        img, img_features = generate_image(lens_mass_kwargs, src_light_kwargs, self.null_psf, self.data_api, lens_mass_model, self.src_light_model)
        img /= np.max(img)
        dmag = -2.5*np.log10(img_features['total_magnification'])
        img_features['magnorms'] = {band: src_light_info[f'magnorm_{bulge_or_disk}_{band}'] + dmag for band in self.bands}
        return img, img_features

    def get_cored_image(self, lens_info, src_light_info, z_lens, z_src, bulge_or_disk):
        """Render a lensed cored sersic

        Note
        ----
        This method is only used to test the total magnification computation

        """
        lens_mass_model = LensModel(['SIE', 'SHEAR_GAMMA_PSI',], cosmo=self.cosmo, z_lens=z_lens, z_source=z_src)
        lens_mass_kwargs = get_lens_params(lens_info, z_src=z_src, cosmo=self.cosmo)
        src_light_kwargs = get_cored_sersic_params(src_light_info, bulge_or_disk=bulge_or_disk) 
        img, img_features = generate_image(lens_mass_kwargs, src_light_kwargs, self.null_psf, self.data_api, lens_mass_model, self.cored_sersic_model)
        img /= np.max(img)
        return img, img_features

def get_unlensed_total_flux_analytical(kwargs_src_light_list, src_light_model):
    """Compute the total flux of unlensed objects

    Parameter
    ---------
    kwargs_src_light_list : list
        list of kwargs dictionaries for the unlensed source galaxy, each with an 'amp' key
    kwargs_ps_list : list
        list of kwargs dictionaries for the unlensed point source (if any), each with an 'amp' key

    Returns
    -------
    float
        the total unlensed flux

    """
    total_flux = 0.0
    for i, kwargs_src in enumerate(kwargs_src_light_list):
        total_flux += src_light_model.total_flux(kwargs_src_light_list, norm=True, k=i)[0]
    return total_flux
        
def get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, image_model):
    """Compute the total flux of the lensed image

    Returns
    -------
    float
        the total lensed flux

    """

    lensed_src_image = image_model.image(kwargs_lens_mass, kwargs_src_light, None, None, lens_light_add=False)
    lensed_total_flux = np.sum(lensed_src_image)
    return lensed_total_flux

def get_unlensed_total_flux_numerical(kwargs_src_light, image_model):
    """Compute the total flux of the unlensed image by rendering the source on a pixel grid

    Returns
    -------
    float
        the total lensed flux

    """

    unlensed_src_image = image_model.image(None, kwargs_src_light, None, None, lens_light_add=False)
    unlensed_total_flux = np.sum(unlensed_src_image)
    return unlensed_total_flux

def generate_image(kwargs_lens_mass, kwargs_src_light, psf_model, data_api, lens_mass_model, src_light_model):
    """Generate the image of a lensed extended source from provided model and model parameters

    Parameters
    ----------
    kwargs_lens_mass : dict
        lens model parameters
    kwargs_src_light : dict
        host light model parameters
    psf_model : lenstronomy PSF object
        the PSF kernel point source map
    data_api : lenstronomy DataAPI object
        tool that handles detector and observation conditions 

    Returns
    -------
    tuple of (np.array, dict)
        the lensed image

    """
    img_features = {}
    kwargs_numerics = {'supersampling_factor': 1}
    image_data = data_api.data_class
    # Instantiate image model
    lensed_image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model, None, None, kwargs_numerics=kwargs_numerics)
    # Compute total magnification
    lensed_total_flux = get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, lensed_image_model)
    img_features['lensed_total_flux'] = lensed_total_flux
    #try: 
    #unlensed_total_flux = get_unlenseD_total_flux_analytical(kwargs_src_light_list, src_light_model)
    unlensed_image_model = ImageModel(image_data, psf_model, None, src_light_model, None, None, kwargs_numerics=kwargs_numerics)
    unlensed_total_flux = get_unlensed_total_flux_numerical(kwargs_src_light, unlensed_image_model) # analytical only runs for profiles that allow analytic integration
    img_features['total_magnification'] = lensed_total_flux/unlensed_total_flux
    img_features['unlensed_total_flux'] = unlensed_total_flux
    #except:
    #    pass
    # Generate image for export
    img = lensed_image_model.image(kwargs_lens_mass, kwargs_src_light, None, None)
    img = np.maximum(0.0, img) # safeguard against negative pixel values
    return img, img_features

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
    e_tmp, lef_tmp = np.loadtxt('data/ell_lef.dat', comments='#', usecols=(0,1), unpack=True)
    interpolated_lambdas = interp1d(e_tmp, lef_tmp, kind='linear')
    return interpolated_lambdas(ellip)

def get_null_psf(pixel_scale):
    """Get a null (delta function) PSF

    Parameters
    ----------
    pixel_scale : float
        arcsec per pixel

    """
    return PSF(psf_type='NONE')

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
    lens_phie_rad = np.pi*(lens_info['phie_lens']/180.0) + 0.5*np.pi # in rad, origin at y-axis
    lens_e1, lens_e2 = param_util.phi_q2_ellipticity(lens_phie_rad, 1 - lens_info['ellip_lens'])
    # Instantiate cosmology-aware models
    lens_cosmo = LensCosmo(z_lens=lens_info['redshift'], z_source=z_src, cosmo=cosmo)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(lens_info['vel_disp_lenscat'])[0]
    lam = get_lambda_factor(lens_info['ellip_lens']) # removed because lenstronomy accepts the spherically-averaged Einstein radius as input
    phi, q = param_util.ellipticity2phi_q(lens_e1, lens_e2)
    gravlens_to_lenstronomy = np.sqrt((1.0 + q**2.0)/(2.0*q)) # factor converting the grav lens ellipticity convention (square average) to lenstronomy's (product average)
    sie_mass = dict(
                      center_x=0.0,
                      center_y=0.0,
                      #s_scale=0.0,
                      theta_E=theta_E*gravlens_to_lenstronomy,
                      e1=lens_e1,
                      e2=lens_e2
                      )
    external_shear = dict(
                          gamma_ext=lens_info['gamma_lenscat'],
                          psi_ext=np.deg2rad(lens_info['phig_lenscat'])
                          )
    #external_shear = dict(
    #                      gamma1=lens_info['shear_1_lenscat'],
    #                      gamma2=lens_info['shear_2_lenscat']
    #                      )
    return [sie_mass, external_shear]

def get_src_light_params(src_light_info, bulge_or_disk='bulge'):
    """Get Sersic parameters into a form Lenstronomy understands

    Parameters
    ----------
    src_light_info : dict
        Sersic host galaxy component (bulge or disk) parameters for a system
    bulge_or_disk : str
        galaxy component ('bulge' or 'disk')

    """
    phi_rad = 0.5*np.deg2rad(src_light_info['position_angle'])
    n_sersic = src_light_info['sindex_{:s}'.format(bulge_or_disk)]
    R_sersic = (src_light_info['major_axis_{:s}'.format(bulge_or_disk)]*src_light_info['minor_axis_{:s}'.format(bulge_or_disk)])**0.5
    q_src = src_light_info['minor_axis_{:s}'.format(bulge_or_disk)]/src_light_info['major_axis_{:s}'.format(bulge_or_disk)]
    e1_src, e2_src = param_util.phi_q2_ellipticity(phi_rad, q_src)

    sersic_host = dict(
                     amp=20.0, # doesn't matter, image gets rescaled anyway
                     n_sersic=n_sersic,
                     R_sersic=R_sersic,
                     center_x=src_light_info['x_src'],
                     center_y=src_light_info['y_src'],
                     e1=e1_src,
                     e2=e2_src,
                     )
    return [sersic_host]

def get_cored_sersic_params(src_light_info, bulge_or_disk='bulge'):
    """Get Sersic parameters into a form Lenstronomy understands

    Parameters
    ----------
    src_light_info : dict
        Sersic host galaxy component (bulge or disk) parameters for a system
    bulge_or_disk : str
        galaxy component ('bulge' or 'disk')

    """
    phi_rad = 0.5*np.deg2rad(src_light_info['position_angle'])
    n_sersic = src_light_info['sindex_{:s}'.format(bulge_or_disk)]
    R_sersic = (src_light_info['major_axis_{:s}'.format(bulge_or_disk)]*src_light_info['minor_axis_{:s}'.format(bulge_or_disk)])**0.5
    q_src = src_light_info['minor_axis_{:s}'.format(bulge_or_disk)]/src_light_info['major_axis_{:s}'.format(bulge_or_disk)]
    e1_src, e2_src = param_util.phi_q2_ellipticity(phi_rad, q_src)

    sersic_host = dict(
                     amp=20.0, # doesn't matter, image gets rescaled anyway
                     n_sersic=n_sersic,
                     R_sersic=R_sersic,
                     center_x=src_light_info['x_src'],
                     center_y=src_light_info['y_src'],
                     e1=e1_src,
                     e2=e2_src,
                     Re=0.1*R_sersic,
                     gamma=0.5,
                     max_R_frac=5.0,
                     alpha=10.0,
                     )
    return [sersic_host]

def lens_model_plot_custom(image, ax, lensModel, kwargs_lens, numPix=500, deltaPix=0.01, sourcePos_x=0, sourcePos_y=0, point_source=False, with_caustics=False):
    """
    Overlay the critical curves and caustics over the provided lensed image

    Parameters
    ----------
    image : np.array
    ax : matplotlib.pyplot.axes

    """
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix)
    data = ImageData(**kwargs_data)
    _coords = data
    _frame_size = numPix * deltaPix
    x_grid, y_grid = data.pixel_coordinates
    lensModelExt = LensModelExtensions(lensModel)
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