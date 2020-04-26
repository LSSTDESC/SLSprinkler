#!/usr/bin/env python
import numpy as np
import os
import argparse
import pylab as pl
import subprocess as sp
import astropy.io.fits as pyfits
import pandas as pd
import scipy.special as ss
import om10_lensing_equations as ole
import sqlite3 as sql
import sys
from lensed_hosts_utils import generate_lensed_host

# Have numpy raise exceptions for operations that would produce nan or inf.
np.seterr(invalid='raise', divide='raise', over='raise')

datadefault = 'truth_tables'
outdefault = 'outputs'

parser = argparse.ArgumentParser(description='The location of the data directory')
parser.add_argument("--datadir", dest='datadir', type=str, default = datadefault,
                    help='Location of data directory (containing truth tables)')
parser.add_argument("--outdir", dest='outdir', type=str, default = outdefault,
                    help='Output location for FITS stamps')
parser.add_argument("--pixel_size", type=float, default=0.01,
                    help='Pixel size in arcseconds')
parser.add_argument("--num_pix", type=int, default=1000,
                    help='Number of pixels in x- or y-direction')

args = parser.parse_args()
datadir = args.datadir
outdir = args.outdir


def load_in_data_agn():

    """
    Reads in catalogs of host galaxy bulge and disk as well as om10 lenses

    Returns:
    -----------
    lens_list: data array for lenses.  Includes t0, x, y, sigma, gamma, e, theta_e
    ahb_purged: Data array for galaxy bulges.  Includes prefix, uniqueId, raPhoSim, decPhoSim, phosimMagNorm
    ahd_purged: Data array for galaxy disks.  Includes prefix, uniqueId, raPhoSim, decPhoSim, phosimMagNorm

    """
    conn = sql.connect(os.path.join(datadir,'host_truth.db'))
    agn_host = pd.read_sql_query("select * from agn_hosts;", conn)

    conn2 = sql.connect(os.path.join(datadir,'lens_truth.db'))
    agn_lens = pd.read_sql_query("select * from agn_lens;", conn2)

    idx = agn_host['image_number'] == 0
    ahb_purged = agn_host[idx]

    lens_list = agn_lens

    return lens_list, ahb_purged


def create_cats_agns(index, hdu_list, ahb_list):
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
    ra_lens_check = df_inner['ra_lens_y'][index]
    dec_lens_check = df_inner['dec_lens_y'][index]
    lid = df_inner['lens_cat_sys_id'][index]
    vd = df_inner['vel_disp_lenscat'][index]
    zd = df_inner['redshift_y'][index]
    ql = 1.0 - df_inner['ellip_lens'][index]
    phi = df_inner['position_angle_y'][index]
    ext_shr = df_inner['gamma_lenscat'][index]
    ext_phi = df_inner['phig_lenscat'][index]

    #----------------------------------------------------------------------------
    lens_cat = {'xl1'        : xl1,
                'xl2'        : xl2,
                'ql'         : ql,
                'vd'         : vd,
                'phl'        : phi,
                'gamma'      : ext_shr,
                'phg'        : ext_phi,
                'zl'         : zd,
                'ximg'       : ximg,
                'yimg'       : yimg,
                'twinklesid' : twinkles_ID,
                'lensid'     : lid,
                'index'      : index,
                'UID_lens'   : UID_lens,
                'Ra_lens'    : Ra_lens,
                'Dec_lens'   : Dec_lens,
                'cat_id'     : cat_id}

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
                   'mag_src_u'      : mag_src_b_u,
                   'mag_src_g'      : mag_src_b_g,
                   'mag_src_r'      : mag_src_b_r,
                   'mag_src_i'      : mag_src_b_i,
                   'mag_src_z'      : mag_src_b_z,
                   'mag_src_y'      : mag_src_b_y,
                   'Reff_src'     : Reff_src_b,
                   'qs'           : qs_b,
                   'phs'          : phs_b,
                   'ns'           : ns_b,
                   'zs'           : zs_b,
                   'sed_src'      : sed_src_b,
                   'components'   : 'bulge'}

    #----------------------------------------------------------------------------
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

    srcsP_disk = {'ys1'          : ys1,
                  'ys2'          : ys2,
                  'mag_src_u'      : mag_src_d_u,
                  'mag_src_g'      : mag_src_d_g,
                  'mag_src_r'      : mag_src_d_r,
                  'mag_src_i'      : mag_src_d_i,
                  'mag_src_z'      : mag_src_d_z,
                  'mag_src_y'      : mag_src_d_y,
                  'Reff_src'     : Reff_src_d,
                  'qs'           : qs_d,
                  'phs'          : phs_d,
                  'ns'           : ns_d,
                  'zs'           : zs_d,
                  'sed_src'      : sed_src_d,
                  'components'   : 'disk'}

    return lens_cat, srcsP_bulge, srcsP_disk


if __name__ == '__main__':

    dsx = args.pixel_size  # pixel size per side, arcseconds
    nnn = args.num_pix  # number of pixels per side
    xi1, xi2 = ole.make_r_coor(nnn, dsx)

    hdulist, ahb = load_in_data_agn()

    #hdulist is the list of lens parameters
    #ahb is the list of galaxy bulge and disk parameters

    message_row = 0
    message_freq = 50
    for i, row in hdulist.iterrows():
        if i >= message_row:
            print ("working on system ", i , "of", max(hdulist.index))
            message_row += message_freq
        try:
            lensP, srcPb, srcPd = create_cats_agns(i, hdulist, ahb)
            generate_lensed_host(xi1, xi2, lensP, srcPb, srcPd, dsx,
                                 outdir, 'agn')
        except RuntimeError as eobj:
            print(eobj)
        sys.stdout.flush()
