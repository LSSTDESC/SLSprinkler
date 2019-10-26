import pandas as pd
import numpy as np

__all__ = ['Sprinkler']

class Sprinkler():

    def __init__(self, 
                 gl_agn_cat, glagn_reader,
                 gl_sne_cat, glsne_reader,
                 gal_cat, gal_reader,
                 avoid_gal_ids=None):

        self.gl_agn_cat = pd.DataFrame(glagn_reader(gl_agn_cat).load_catalog())
        self.gl_sne_cat = pd.DataFrame(glsne_reader(gl_sne_cat).load_catalog())
        self.gal_cat = pd.DataFrame(gal_reader(gal_cat).load_catalog())

    def match_agn(self, gal_z, gal_i_mag):

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

    def sprinkle(self):

        agn_gals = self.gal_cat.query('magnorm_agn > -99')
        agn_match_idx = self.match_agn(agn_gals['redshift'].values,
                                       agn_gals['mag_i_agn'].values)
        print(agn_match_idx)

        return