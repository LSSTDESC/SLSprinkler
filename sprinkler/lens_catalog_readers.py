"""
Readers for strongly lensed systems
"""

import numpy as np
import pandas as pd
from astropy.io import fits

__all__ = ['OM10Reader', 'GoldsteinSNeCatReader']

class OM10Reader():

    """
    Reader for OM10 type catalog that has already
    matched to catalog for additional galaxy properties
    for the lensing galaxy like reff, sed filename,
    lens AV, lens RV and magnitude.
    """

    def __init__(self, filename):

        self.filename = filename

        self.config = {
            'system_id':'LENSID',
            'z_src':'ZSRC',
            'mag_i_src':'MAGI_IN',
            'n_img':'NIMG',
            'x_img':'XIMG',
            'y_img':'YIMG',
            't_delay_img':'DELAY',
            'magnification_img':'MAG',
            'sed_lens':'lens_sed',
            'magnorm_lens':'sed_magNorm',
            'z_lens':'ZLENS',
            'reff_lens':'REFF',
            'ellip_lens':'ELLIP',
            'phie_lens':'PHIE',
            'av_lens':'lens_av',
            'rv_lens':'lens_rv',
            'vel_disp_lens':'VELDISP',
            'gamma':'GAMMA',
            'phi_gamma':'PHIG'
        }

    def load_catalog(self):

        om10_cat = fits.open(self.filename)[1]
        lensed_agn_cat = {}

        for key, val in self.config.items():
            cat_vals = om10_cat.data[val]
            cat_vals = cat_vals.byteswap().newbyteorder()
            if len(np.shape(cat_vals)) == 1:
                lensed_agn_cat[key] = cat_vals
            else:
                lensed_agn_cat[key] = list(cat_vals)

        return lensed_agn_cat

class GoldsteinSNeCatReader():

    def __init__(self, filename):

        self.filename = filename

        self.config = {
            'system_id':'sysno',
            'z_src':'zs',
            'n_img':'n_img',
            'x_img':'x_img',
            'y_img':'y_img',
            't_delay_img':'t_delay_img',
            'magnification_img':'magnification_img',
            'MB_host':'MB',
            'type_host':'host_type',
            'sed_lens':'lensgal_sed',
            'magnorm_lens':'magnorm_lens',
            'z_lens':'zl',
            'reff_lens':'lensgal_reff',
            'ellip_lens':'lensgal_ellip',
            'phie_lens':'theta_e',
            'av_lens':'lens_av',
            'rv_lens':'lens_rv',
            'vel_disp_lens':'sigma',
            'gamma':'gamma',
            'phi_gamma':'theta_gamma',
            'x0':'x0',
            'x1':'x1',
            'c':'c'
        }

    def merge_catalog(self, df_sys, df_img):

        img_pivot = df_img.pivot(index='sysno', columns='imno')
        img_df = pd.DataFrame([])
        img_df['sysno'] = img_pivot.index.values
        img_df['n_img'] = np.sum(~np.isnan(img_pivot['td']), axis=1).values
        img_df['x_img'] = list(img_pivot['x'].values)
        img_df['y_img'] = list(img_pivot['y'].values)
        img_df['t_delay_img'] = list(img_pivot['td'].values)
        img_df['magnification_img'] = list(img_pivot['mu'].values)

        df_merged = df_sys.merge(img_df, on='sysno')

        return df_merged

    def merge_magnorms(self, df_merged):

        lensgal_magnorm = df_merged[['lensgal_magnorm_u',
                                     'lensgal_magnorm_g',
                                     'lensgal_magnorm_r',
                                     'lensgal_magnorm_i',
                                     'lensgal_magnorm_z',
                                     'lensgal_magnorm_y']].values
        
        df_merged['magnorm_lens'] = list(lensgal_magnorm)

        return df_merged

    def load_catalog(self):

        sne_systems = pd.read_hdf(self.filename, key='system')
        sne_images = pd.read_hdf(self.filename, key='image')
        
        sne_merged = self.merge_catalog(sne_systems, sne_images)

        sne_merged = self.merge_magnorms(sne_merged)

        lensed_sne_cat = {}

        for key, val in self.config.items():
            cat_vals = sne_merged[val]
            lensed_sne_cat[key] = cat_vals

        return lensed_sne_cat