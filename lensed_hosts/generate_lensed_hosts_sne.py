#!/usr/bin/env python
import numpy as np
import sys
import os
import pylab as pl
import argparse
import subprocess as sp
import astropy.io.fits as pyfits
import pandas as pd
import scipy.special as ss
import om10_lensing_equations as ole
import sqlite3 as sql
from lensed_hosts_utils import generate_lensed_host

# Have numpy raise exceptions for operations that would produce nan or inf.
np.seterr(invalid='raise', divide='raise', over='raise')

datadefault = 'truth_tables'
outdefault = 'outputs'

parser = argparse.ArgumentParser(description='The location of the desired output directory')
parser.add_argument("--datadir", dest='datadir', type=str, default = datadefault,
                    help='Location of input truth tables')
parser.add_argument("--outdir", dest='outdir', type=str, default = outdefault,
                    help='Output location for FITS stamps')
parser.add_argument("--pixel_size", type=float, default=0.01,
                    help='Pixel size in arcseconds')
parser.add_argument("--num_pix", type=int, default=1000,
                    help='Number of pixels in x- and y-directions')
parser.add_argument("--seed", type=int, default=42,
                    help='Seed for random draw of galaxy locations.')
args = parser.parse_args()
datadir = args.datadir
outdir = args.outdir


def random_location(Reff_src, qs, phs, ns, rng=None):
    """Sample a random (x, y) location from the surface brightness
    profile of the galaxy. The input parameters are Sersic parameters for the host galaxy.
    Parameters:
    -----------
    Reff_src: float
        the effective radius in arcseconds, the radius within which half of the light is contained
    qs: float
        axis ratio of the source, b/a
    phs: float
        position angle of the galaxy in degrees
    ns: int
        Sersic index
    rng: numpy.random.RandomState [None]
        RandomState object to use for generating random draws from [0, 1).
        If None, then create a RandomState with default seeding.

    Returns:
    -----------
    dx: horizontal coordinate of random location (pixel coordinates)
    dy: vertical coordinate of random location (pixel coordinates)
    """
    if rng is None:
        rng = np.random.RandomState()

    phs_rad = np.deg2rad(phs-90)

    bn = ss.gammaincinv(2. * ns, 0.5)
    z = rng.random_sample()
    x = ss.gammaincinv(2. * ns, z)
    R = (x / bn)**ns * Reff_src
    theta = rng.random_sample() * 2 * np.pi

    xp, yp = R * np.cos(theta), R * np.sin(theta)
    xt = xp * np.sqrt(qs)
    yt = yp / np.sqrt(qs)
    dx, dy = np.linalg.solve([[np.cos(phs_rad), np.sin(phs_rad)],
                             [-np.sin(phs_rad), np.cos(phs_rad)]],
                             [xt, yt])
    return dx, dy


def check_random_locations():
    """Defines a random location to compare to"""

    npoints = 100000
    Reff_disk = 0.2
    qs_disk = 0.3
    phs_disk = 8.
    ns_disk = 1.0

    x_d = np.zeros(npoints)
    y_d = np.zeros(npoints)

    for i in range(npoints):
        x_d[i], y_d[i] = random_location(Reff_disk, qs_disk, phs_disk, ns_disk)


    bsz = 5.0
    nnn = 1000  # number of pixels per side
    dsx = bsz/nnn
    xi1, xi2 = ole.make_r_coor(nnn, dsx)

    src_disk = ole.sersic_2d(xi1,xi2,0.0,0.0,Reff_disk,qs_disk,phs_disk,ns_disk)
    src_disk_norm = src_disk/(np.sum(src_disk)*dsx*dsx)

    src_disk_px = np.sum(src_disk, axis=1)
    src_disk_norm_px = src_disk_px/(np.sum(src_disk_px)*dsx)

    src_disk_py = np.sum(src_disk, axis=0)
    src_disk_norm_py = src_disk_py/(np.sum(src_disk_py)*dsx)

    xsteps = xi1[:,0]
    #---------------------------------------------------------------

    from matplotlib.ticker import NullFormatter

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    pl.figure(1, figsize=(8, 8))

    axScatter = pl.axes(rect_scatter)
    axHistx = pl.axes(rect_histx)
    axHisty = pl.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x_d, y_d)
    axScatter.contour(xi1, xi2, src_disk, colors=['k',])

    # now determine nice limits by hand:
    binwidth = 0.02
    xymax = max(np.max(np.abs(x_d)), np.max(np.abs(y_d)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x_d, bins=bins, density=1)
    axHistx.plot(xsteps, src_disk_norm_px, 'k-')

    axHisty.hist(y_d, bins=bins, density=1,orientation='horizontal')
    axHisty.plot(src_disk_norm_py, xsteps, 'k-')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    return 0

def load_in_data_sne():
    """
    Reads in catalogs of host galaxy bulge and disk as well as om10 lenses

    Returns:
    -----------
    slc_purged: data array for lenses.  Includes t0, x, y, sigma, gamma, e, theta_e
    shb_purged: Data array for galaxy bulges.  Includes prefix, uniqueId, raPhoSim, decPhoSim, phosimMagNorm
    shd_purged: Data array for galaxy disks.  Includes prefix, uniqueId, raPhoSim, decPhoSim, phosimMagNorm

    """

    conn = sql.connect(os.path.join(datadir,'host_truth.db'))
    sne_host = pd.read_sql_query("select * from sne_hosts;", conn)

    conn2 = sql.connect(os.path.join(datadir,'lens_truth.db'))
    sne_lens = pd.read_sql_query("select * from sne_lens;", conn2)

    idx = sne_host['image_number'] == 0
    shb_purged = sne_host[idx]

    slc_purged = sne_lens

    return slc_purged, shb_purged


def create_cats_sne(index, hdu_list, ahb_list, rng=None):
    """
    Takes input catalogs and isolates lensing parameters as well as ra and dec of lens
    Parameters:
    -----------
    index: int
        Index for pandas data frame
    hdu_list:
        row of data frame that contains lens parameters
    ahb_list:
        row of data frame that contains lens galaxy parameters for the galactic bulge
    rng:
        numpy.random.RandomState object used to generate galaxy position
        draws.

    Returns:
    -----------
    lens_cat: Data array that includes lens parameters
    srcsP_bulge: Data array that includes parameters for galactic bulge
    srcsP_disk: Data array that includes parameters for galactic disk

    """

    df_inner = pd.merge(ahb_list, hdu_list, on='lens_cat_sys_id', how='inner')

#    for col in df_inner.columns:
#        print(col)
    dc2_sys_id_tokens = df_inner['dc2_sys_id_x'][index].split('_')
    UID_lens = '_'.join((dc2_sys_id_tokens[0], 'host', dc2_sys_id_tokens[1],
                         str(df_inner['image_number'][index])))
    twinkles_ID = UID_lens
    cat_id = df_inner['unique_id_x'][index]
    Ra_lens = df_inner['ra_lens_x'][index]
    Dec_lens = df_inner['dec_lens_x'][index]
    ys1 = df_inner['x_src'][index]
    ys2 = df_inner['y_src'][index]
    ximg = df_inner['x_img'][index]
    yimg = df_inner['y_img'][index]
    xl1 = 0.0
    xl2 = 0.0
    lid = df_inner['lens_cat_sys_id'][index]
    vd = df_inner['vel_disp_lenscat'][index]   # needed from OM10
    zd = df_inner['redshift_y'][index] 
#    ql = 1.0 - df_inner['ellip_lenscat'][index]
    ql = 1.0 - df_inner['ellip_lens'][index]
    phi= df_inner['position_angle_y'][index] 
    ext_shr = df_inner['gamma_lenscat'][index]
    ext_phi = df_inner['phig_lenscat'][index]

    if not (np.isfinite(df_inner['x_src'][index]) and
            np.isfinite(df_inner['y_src'][index])):
        raise RuntimeError(f'x_src or y_src is not finite for lens id {lid}')

    #----------------------------------------------------------------------------
    lens_cat = {'xl1'        : xl1,
                'xl2'        : xl2,
                'ql'         : ql,
                'vd'         : vd,
                'phl'        : phi,
                'gamma'      : ext_shr,
                'phg'        : ext_phi,
                'zl'         : zd,
                'twinklesid' : twinkles_ID,
                'lensid'     : lid,
                'index'      : index,
                'UID_lens'   : UID_lens,
                'Ra_lens'    : Ra_lens,
                'Dec_lens'   : Dec_lens,
                'cat_id'     : cat_id}

    #----------------------------------------------------------------------------
    bands = 'ugrizy'
    for galtype in ('disk', 'bulge'):
        if any([not np.isfinite(df_inner[f'magnorm_{galtype}_{band}'][index])
                for band in bands]):
            raise RuntimeError('non-finite magnorm values in sne_hosts table '
                               f'for lens id {lid}')

    mag_src_d_u = df_inner['magnorm_disk_u'][index]
    mag_src_d_g = df_inner['magnorm_disk_g'][index]
    mag_src_d_r = df_inner['magnorm_disk_r'][index]
    mag_src_d_i = df_inner['magnorm_disk_i'][index]
    mag_src_d_z = df_inner['magnorm_disk_z'][index]
    mag_src_d_y = df_inner['magnorm_disk_y'][index]
    qs_d = df_inner['minor_axis_disk'][index]/df_inner['major_axis_disk'][index]
    Reff_src_d = np.sqrt(df_inner['minor_axis_disk'][index]*df_inner['major_axis_disk'][index])
    phs_d = df_inner['position_angle_x'][index]
    ns_d = df_inner['sindex_disk'][index]
    zs_d = df_inner['redshift_x'][index]
    sed_src_d = df_inner['sed_disk_host'][index]

    dys2, dys1 = random_location(Reff_src_d, qs_d, phs_d, ns_d, rng)
    ys1 = df_inner['x_src'][index] - dys1    # needed more discussion
    ys2 = df_inner['y_src'][index] - dys2    # needed more discussion
    if np.abs(ys1) > 5 or np.abs(ys2) > 5:
        print(f'ys1, ys2: {ys1:.3f}  {ys2:.3f}')

    srcsP_disk = {'ys1'          : ys1,
                  'ys2'          : ys2,
                  'mag_src_u'  : mag_src_d_u,
                  'mag_src_g'  : mag_src_d_g,
                  'mag_src_r'  : mag_src_d_r,
                  'mag_src_i'  : mag_src_d_i,
                  'mag_src_z'  : mag_src_d_z,
                  'mag_src_y'  : mag_src_d_y,
                  'Reff_src'     : Reff_src_d,
                  'qs'           : qs_d,
                  'phs'          : phs_d,
                  'ns'           : ns_d,
                  'zs'           : zs_d,
                  'sed_src'      : sed_src_d,
                  'lensid'       : lid,
                  'components'   : 'disk'}

    #----------------------------------------------------------------------------

    mag_src_b_u = df_inner['magnorm_bulge_u'][index]
    mag_src_b_g = df_inner['magnorm_bulge_g'][index]
    mag_src_b_r = df_inner['magnorm_bulge_r'][index]
    mag_src_b_i = df_inner['magnorm_bulge_i'][index]
    mag_src_b_z = df_inner['magnorm_bulge_z'][index]
    mag_src_b_y = df_inner['magnorm_bulge_y'][index]
    qs_b = df_inner['minor_axis_bulge'][index]/df_inner['major_axis_bulge'][index]
    Reff_src_b = np.sqrt(df_inner['minor_axis_bulge'][index]*df_inner['major_axis_bulge'][index])
    phs_b = df_inner['position_angle_x'][index]
    ns_b = df_inner['sindex_bulge'][index]
    zs_b = df_inner['redshift_x'][index]
    sed_src_b = df_inner['sed_bulge_host'][index]

    srcsP_bulge = {'ys1'          : ys1,
                   'ys2'          : ys2,
                   'mag_src_u'  : mag_src_b_u,
                   'mag_src_g'  : mag_src_b_g,
                   'mag_src_r'  : mag_src_b_r,
                   'mag_src_i'  : mag_src_b_i,
                   'mag_src_z'  : mag_src_b_z,
                   'mag_src_y'  : mag_src_b_y,
                   'Reff_src'     : Reff_src_b,
                   'qs'           : qs_b,
                   'phs'          : phs_b,
                   'ns'           : ns_b,
                   'zs'           : zs_b,
                   'sed_src'      : sed_src_b,
                   'lensid'       : lid,
                   'components'   : 'bulge'}

    return lens_cat, srcsP_bulge, srcsP_disk


if __name__ == '__main__':

    dsx = args.pixel_size  # pixel size per side, arcseconds
    nnn = args.num_pix  # number of pixels per side
    xi1, xi2 = ole.make_r_coor(nnn, dsx)

    rng = np.random.RandomState(args.seed)

    hdulist, ahb = load_in_data_sne()

    message_row = 0
    message_freq = 50
    for i, row in hdulist.iterrows():
        if i >= message_row:
            print ("working on system ", i , "of", max(hdulist.index))
            message_row += message_freq
        try:
            lensP, srcPb, srcPd = create_cats_sne(i, hdulist, ahb, rng)
            generate_lensed_host(xi1, xi2, lensP, srcPb, srcPd, dsx,
                                 outdir, 'sne')
        except RuntimeError as eobj:
            print(eobj)
        sys.stdout.flush()
