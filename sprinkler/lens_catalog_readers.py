"""
Readers for strongly lensed systems
"""

import numpy as np
from astropy.io import fits

__all__ = ['OM10Reader']

class OM10Reader():

    def __init__(self, filename):

        self.filename = filename

        self.config = {
            'system_id':'LENSID',
            'z_src':'ZSRC',
            'n_img':'NIMG',
            'x_img':'XIMG',
            'y_img':'YIMG',
            't_delay_img':'DELAY',
            'magnification_img':'MAG',
            'sed_lens':'lens_sed',
            'z_lens':'ZLENS',
            'reff_lens':'REFF',
            'ellip_lens':'ELLIP',
            'phie_lens':'PHIE',
            'av_lens':'lens_av',
            'rv_lens':'lens_rv'
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