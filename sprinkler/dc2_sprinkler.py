import pandas as pd
import numpy as np
from .base_sprinkler import BaseSprinkler

__all__ = ['DC2Sprinkler']


class DC2Sprinkler(BaseSprinkler):

    def match_agn(self, gal_cat):

        gal_z = gal_cat['redshift'].values
        gal_i_mag = gal_cat['mag_i_agn'].values

        # search the OM10 catalog for all sources +- 0.1 dex in redshift
        # and within .25 mags of the AGN source
        lens_candidate_idx = []
        for gal_z_on, gal_i_mag_on in zip(gal_z, gal_i_mag):
            w = np.where((np.abs(np.log10(self.gl_agn_cat['z_src']) -
                                 np.log10(gal_z_on)) <= 0.1) &
                         (np.abs(self.gl_agn_cat['mag_i_src'] -
                                 gal_i_mag_on) <= .25))
            lens_candidate_idx.append(w[0])

        return lens_candidate_idx

    def match_sne(self, gal_cat):

        gal_z = gal_cat['redshift'].values
        gal_type = gal_cat['gal_type'].values

        # search the SNe catalog for all sources +- 0.1 dex in redshift
        # and with matching type
        lens_candidate_idx = []
        for gal_z_on, gal_type_on in zip(gal_z, gal_type):                                                                                                                              
            w = np.where((np.abs(np.log10(self.gl_sne_cat['z_src']) -
                                 np.log10(gal_z_on)) <= 0.1) &
                         (self.gl_sne_cat['type_host'] == gal_type_on))
            lens_candidate_idx.append(w[0])

        return lens_candidate_idx

    def agn_density(self, agn_gal_row):

        return 0.1

    def sne_density(self, sne_gal_row):

        return 1.0

    def output_lens_galaxy_truth(self, matched_gal_cat):

        return

    def output_truth_catalogs(self, matched_gal_cat,
                              matched_agn_cat,
                              matched_sne_cat):

        return

    def generate_matched_catalogs(self):

        agn_hosts, agn_systems = self.sprinkle_agn()
        sne_hosts, sne_systems = self.sprinkle_sne()

        return
