import os
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
#--------------------------------------------------------------------
from astropy.cosmology import WMAP7 as p15
"""I import WMAP7 cosmology (http://docs.astropy.org/en/stable/cosmology/) as it is the cosmology being used for DC2.  
It uses H0 = 70.4, Omega_M = 0.272 and flat universe."""

__all__ = ['Dc', 'Dc2', 're_sv','e2le', 'make_r_coor', 'alphas_sie', 'sersic_2d']

vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
apr =  206264.8        # 1/1^{''}

#data_dir = os.path.join(os.environ['SIMS_GCRCATSIMINTERFACE_DIR'], 'data')
data_dir = 'data'

def Dc(z):
    res = p15.comoving_distance(z).value*p15.h
    return res
    """Dc(z) returns comoving distance at a particular redshift z"""
    """Parameters
    ----------
    z: float
    	redshift
    Returns
    ----------
    res: Comoving distance at this redshift.  """

def Dc2(z1,z2):
    Dcz1 = (p15.comoving_distance(z1).value*p15.h)
    Dcz2 = (p15.comoving_distance(z2).value*p15.h)
    res = Dcz2-Dcz1+1e-8
    return res
    """Dc2(z1,z2) returns comoving distance between two objects at different redshifts.  It accepts z1
	and z2 as input, which are redshifts of lens and source, respectively."""
    """Parameters
    ----------
    z1: float
	lens redshift
    z2: float
	source redshift
    Returns
    ----------
    res: Comoving distance between lens and source."""
	
def re_sv(sigmav, z1, z2):
    res = 4.0*np.pi*(sigmav/vc)**2.0*Dc2(z1, z2)/Dc(z2)*apr
    return res
    """Takes in sigmav (velocity dispersion) and lens (z1) and source (z2) redshifts and returns Einstein radius, re_sv.  
       The speed of light (vc) is given at the beginning of this file. 
       This is how Einstein radius depends on velocity dispersion in a singular isothermal sphere model."""
    """Parameters
    ----------
    sigmav: float
        velocity dispersion in km/s
    z1: float
        redshift of lens
    z2: float
        redshift of source
    Returns
    ----------
    res: Einstein radius in arcsec"""

#--------------------------------------------------------------------

def e2le(e_in):

    e_tmp,lef_tmp = np.loadtxt(os.path.join(data_dir, "ell_lef.dat"),
                               comments='#',usecols=(0,1),unpack=True)
    f1 = interp1d(e_tmp, lef_tmp, kind='linear')

    return f1(e_in)

    """This routine takes in a particular ellipticity and uses a lookup table to find a
    scale factor due to projection of ellipsoid.  This scale factor is used in alphas_sie and kappa_sie below.
    It corresponds to the Lambda(e) parameter given in Oguri and Marshall (10), Eq. 1."""
    """Parameters
    ----------
    e_in: float
    	ellipticity (1-axis_ratio for lens)
    Returns
    ----------
    f1: scale factor """

def make_r_coor(nc,dsx):
    bsz = nc*dsx
    x1 = np.linspace(0,bsz-dsx,nc)-bsz/2.0+dsx/2.0
    x2 = np.linspace(0,bsz-dsx,nc)-bsz/2.0+dsx/2.0
    x2,x1 = np.meshgrid(x1,x2)
    return x1,x2
    """Create a grid for the output image of the lensed host."""   
    """Parameters
    ----------
    nc: int
        number of pixels per side in FITS image of lensed host   
    dsx: float
        pixel scale, arcseconds per pixel

    Returns
    ---------- 
    x1: array of horizontal coordinates, in pixel coordinates
    x2: array of vertical coordinates, in pixel coordinates """

def alphas_sie(x0, y0, theta, ql, re, le, ext_shears, ext_angle, ext_kappa, x, y):   
    tr = np.pi * (theta / 180.0)   + np.pi / 2.0

    sx = x - x0
    sy = y - y0

    cs = np.cos(tr)
    sn = np.sin(tr)

    sx_r = sx * cs + sy * sn
    sy_r = -sx * sn + sy * cs

    eql = np.sqrt(ql / (1.0 - ql**2.0))
    psi = np.sqrt(sx_r**2.0 * ql + sy_r**2.0 / ql)
    dx_tmp = (re * eql * np.arctan( sx_r / psi / eql))
    dy_tmp = (re * eql * np.arctanh(sy_r / psi / eql))

    dx = dx_tmp * cs - dy_tmp * sn
    dy = dx_tmp * sn + dy_tmp * cs

    # external shear
    tr2 = np.pi * (ext_angle / 180.0)
    cs2 = np.cos(2.0 * tr2)
    sn2 = np.sin(2.0 * tr2)
    dx2 = ext_shears * (cs2 * sx + sn2 * sy)
    dy2 = ext_shears * (sn2 * sx - cs2 * sy)

    # external kappa
    dx3 = ext_kappa * sx
    dy3 = ext_kappa * sy
    return dx*le + dx2 + dx3, dy*le + dy2 + dy3

    """Uses Singular Isothermal Ellipsoid lens model.  Ultimately produces new positions for each input pixel.  (Alpha is lensed position.)"""   
    """Parameters
    ----------
    x0: float
        x position of the lens, in pixel coordinates
    y0: float
        y position of the lens, in pixel coordinates
    theta: float
        position angle of the lens, degree 
    ql: float
        axis ratio b/a
    re: float
        Einstein radius of lens, arcseconds.
    le: float
        scale factor due to projection of ellipsoid
    ext_shears: float
        external shear
    ext_angle: float
        position angle of external shear
    ext_kappa: float
        external convergence
    x: float
        array x position, generated by make_r_coors, in pixel coordinates
    y: float
        array y-position, generated by make_c_coors, in pixel coordinates

    Returns
    ---------- 
    ai1, ai2: lensed positions in x and y (in pixel coordinates) for the two lensing programs, generate_lensed_hosts_***.py """

    # Rotate regular grids
    
#--------------------------------------------------------------------
def xy_rotate(x, y, xcen, ycen, phi):
    phirad = np.deg2rad(phi)
    xnew = (x-xcen)*np.cos(phirad)+(y-ycen)*np.sin(phirad)
    ynew = (y-ycen)*np.cos(phirad)-(x-xcen)*np.sin(phirad)
    return (xnew,ynew)

"""Rotate regular grids """
    
"""Parameters
   ----------
   x: float array
    original x-positions (pixel coordinates)
   y: float array
    original y-positions (pixel coordinates)
   xcen: float
    x-position to rotate around (pixel coordinates)
   ycen: float
    y-position to rotate around (pixel coordinates)
   phi: float
    rotation angle (degrees)

   Returns
   ----------  
   xnew: rotated x-positions (pixel coordinates)
   ynew: rotated y-positions (pixel coordinates)      """

#--------------------------------------------------------------------

def sersic_2d(xi1,xi2,xc1,xc2,Reff_arc,ql,pha,ndex):
    bn = 2.0*ndex-1/3.0+0.009876/ndex
    (xi1new,xi2new) = xy_rotate(xi1, xi2, xc1, xc2, pha)
    R_scale = np.sqrt((xi1new**2)*ql+(xi2new**2)/ql)/Reff_arc
    img = np.exp(-bn*((R_scale)**(1.0/ndex)-1.0))
    R_in = 0.1 # in the units of Reff_arc
    img_max = np.exp(-bn*((R_in)**(1.0/ndex)-1.0))
    img[np.where(R_scale<R_in)] = img_max
    res = img/img_max
    R_out= 5.0 # in the units of Reff_arc
    res[np.where(R_scale>R_out)] = 0.0
    return res
    """Produces a 2-D Sersic profile, Peak = 1.0"""   

    """Parameters
    ----------
    xi1:float array
        horizontal position for Sersic profile (pixel coordinates)
    xi2:float array
        vertical position for Sersic profile (pixel coordinates)
    xc1:float
        horizontal position of source (pixel coordinates) 
    xc2: float
        vertical position of source (pixel coordinates)
    Reff_arc: float
        the effective radius in arcseconds, the radius within which half of the light is contained
    ql: float
        axis ratio, b/a
    pha: float
        position angle of the galaxy in degrees
    ndex: int
        Sersic index 

    Returns
    ----------
    res: array of flux density predicted by Sersic profile. To find magnitude, sum all values in this array 
    and then magnitude is = - 2.5*np.log(np.sum(res)/np.sum(reference), where res is the result of sersic_2d     for some object and reference is the result of sersic_2d for some reference object               """
#--------------------------------------------------------------------
