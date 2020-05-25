"""
Readers for galaxy catalogs to match to lens systems
"""

import numpy as np
import pandas as pd
import GCRCatalogs
import sqlite3

__all__ = ['DC2Reader']

class DC2Reader():

    """
    Reader for cosmoDC2 galaxies supplemented with AGN
    and SED information.
    """

    def __init__(self, catalog_version):

        self.catalog_version = catalog_version

        # The columns we need to query
        self.quantity_list = [
            'galaxy_id',
            'ra',
            'dec',
            'redshift_true',
            'shear_1',
            'shear_2_phosim',
            'convergence',
            'position_angle_true',
            'size_true',
            'size_minor_true',
            'size_disk_true',
            'size_minor_disk_true',
            'size_bulge_true',
            'size_minor_bulge_true',
            'ellipticity_true',
            'sersic_disk',
            'sersic_bulge',
            'stellar_mass_bulge',
            'stellar_mass',
            'totalStarFormationRate',
            'morphology/spheroidHalfLightRadius',
            'morphology/spheroidHalfLightRadiusArcsec',
            'mag_true_r_lsst',
            'mag_true_i_lsst'
        ]

    def load_galaxy_catalog(self, catalog_filters):

        catalog = GCRCatalogs.load_catalog(self.catalog_version)
        dc2_galaxies = catalog.get_quantities(self.quantity_list,
                                              catalog_filters)
        dc2_galaxies_df = pd.DataFrame(dc2_galaxies)

        return dc2_galaxies_df

    def trim_catalog(self, full_lens_df):

        # Keep only "elliptical" galaxies. Use bulge/total mass ratio as proxy.
        trim_idx_ellip = np.where(full_lens_df['stellar_mass_bulge']/
                                  full_lens_df['stellar_mass'] > 0.99)[0]
        trim_lens_catalog = full_lens_df.iloc[trim_idx_ellip].reset_index(drop=True)

        return trim_lens_catalog

    def load_agn_catalog(self, agn_db_file, agn_trim_query):

        conn = sqlite3.connect(agn_db_file)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

        agn_df = pd.read_sql_query("SELECT * FROM agn_params", conn)
        trim_agn_df = agn_df.query(agn_trim_query).reset_index(drop=True)

        return trim_agn_df
