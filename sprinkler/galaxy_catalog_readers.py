"""
Readers for strongly lensed systems
"""

import numpy as np
import pandas as pd
import pickle

__all__ = ['DC2Reader']

class DC2Reader():

    """
    Reader for cosmoDC2 galaxies supplemented with AGN
    and SED information.
    """

    def __init__(self, filename):

        self.filename = filename

        self.config = {
            'galaxy_id':'galaxy_id',
            'ra':'ra',
            'dec':'dec',
            'redshift':'redshift',
            'gamma_1':'shear_1',
            'gamma_2':'shear_2_phosim',
            'kappa':'convergence',
            'av_internal_disk':'disk_av',
            'av_internal_bulge':'bulge_av',
            'rv_internal_disk':'disk_rv',
            'rv_internal_bulge':'bulge_rv',
            'av_mw':'av_mw',
            'rv_mw':'rv_mw',
            'semi_major_axis_disk':'size_disk_true',
            'semi_major_axis_bulge':'size_bulge_true',
            'semi_minor_axis_disk':'size_minor_disk_true',
            'semi_minor_axis_bulge':'size_minor_bulge_true',
            'position_angle':'position_angle_true',
            'magnorm_u_bulge':'bulge_magnorm_u',
            'magnorm_g_bulge':'bulge_magnorm_g',
            'magnorm_r_bulge':'bulge_magnorm_r',
            'magnorm_i_bulge':'bulge_magnorm_i',
            'magnorm_z_bulge':'bulge_magnorm_z',
            'magnorm_y_bulge':'bulge_magnorm_y',
            'magnorm_u_disk':'disk_magnorm_u',
            'magnorm_g_disk':'disk_magnorm_g',
            'magnorm_r_disk':'disk_magnorm_r',
            'magnorm_i_disk':'disk_magnorm_i',
            'magnorm_z_disk':'disk_magnorm_z',
            'magnorm_y_disk':'disk_magnorm_y',
            'flux_u_bulge':'bulge_fluxes_u',
            'flux_g_bulge':'bulge_fluxes_g',
            'flux_r_bulge':'bulge_fluxes_r',
            'flux_i_bulge':'bulge_fluxes_i',
            'flux_z_bulge':'bulge_fluxes_z',
            'flux_y_bulge':'bulge_fluxes_y',
            'flux_u_disk':'disk_fluxes_u',
            'flux_g_disk':'disk_fluxes_g',
            'flux_r_disk':'disk_fluxes_r',
            'flux_i_disk':'disk_fluxes_i',
            'flux_z_disk':'disk_fluxes_z',
            'flux_y_disk':'disk_fluxes_y',
            'magnorm_agn':'magNorm_agn',
            'varParamStr_agn':'varParamStr_agn'
        }

    def merge_labelled_columns(self, df_merged, label):

        labelled_columns = ['%s_u' %label,
                            '%s_g' %label,
                            '%s_r' %label,
                            '%s_i' %label,
                            '%s_z' %label,
                            '%s_y' %label]

        label_array = df_merged[labelled_columns].values
        
        df_merged[label] = list(label_array)

        df_columns = [x for x in df_merged.columns if x not in labelled_columns]

        return df_merged

    def load_catalog(self):

        with open(self.filename, 'rb') as f:
            dc2_cat = pickle.load(f)
        
        dc2_cat = self.merge_labelled_columns(dc2_cat, 'disk_magnorm')
        dc2_cat = self.merge_labelled_columns(dc2_cat, 'bulge_magnorm')
        dc2_cat = self.merge_labelled_columns(dc2_cat, 'disk_fluxes')
        dc2_cat = self.merge_labelled_columns(dc2_cat, 'bulge_fluxes')

        dc2_cat_dict = {}

        for key, val in self.config.items():
            cat_vals = dc2_cat[val]
            dc2_cat_dict[key] = cat_vals

        return dc2_cat_dict
