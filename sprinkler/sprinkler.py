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

        self.avoid_gal_ids = avoid_gal_ids

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

    def agn_density(self, agn_gal_row):

        return 1.0

    def sprinkle_agn(self, agn_density=1.0, rand_state=None):

        if rand_state is None and type(rand_state) != int:
            rand_state = np.random.RandomState(49)
        elif rand_state is None and type(rand_state) == int:
            rand_state = np.random.RandomState(rand_state)

        agn_gals = self.gal_cat.query('magnorm_agn > -99')
        agn_ids = agn_gals['galaxy_id'].values
        if self.avoid_gal_ids is not None:
            agn_gals = agn_gals.iloc[[x for x in np.arange(len(agn_ids))
                                      if agn_ids[x] not in self.avoid_gal_ids]]
        agn_match_idx = self.match_agn(agn_gals['redshift'].values,
                                       agn_gals['mag_i_agn'].values)

        sprinkled_agn_gal_rows = []
        sprinkled_gl_agn_cat_rows = []
        for i, agn_cat_idx in list(enumerate(agn_match_idx)):

            agn_idx_keep = [x for x in agn_cat_idx
                            if x not in sprinkled_gl_agn_cat_rows]

            if len(agn_idx_keep) == 0:
                continue

            agn_density = self.agn_density(agn_gals.iloc[i])

            density_draw = rand_state.uniform()
            if density_draw <= agn_density:
                sprinkled_gl_agn_cat_rows.append(
                    rand_state.choice(agn_idx_keep))
                sprinkled_agn_gal_rows.append(i)

        agn_gals = agn_gals.iloc[sprinkled_agn_gal_rows]
        agn_sys_cat = self.gl_agn_cat.iloc[sprinkled_gl_agn_cat_rows]

        return agn_gals, agn_sys_cat

    def match_sne(self, gal_z, gal_type):

        # search the SNe catalog for all sources +- 0.1 dex in redshift
        # and with matching type
        lens_candidate_idx = []
        for gal_z_on, gal_type_on in zip(gal_z, gal_type):                                                                                                                              
            w = np.where((np.abs(np.log10(self.gl_sne_cat['z_src']) -
                                 np.log10(gal_z_on)) <= 0.1) &
                         (self.gl_sne_cat['type_host'] == gal_type_on))
            lens_candidate_idx.append(w[0])

        return lens_candidate_idx

    def sne_density(self, sne_gal_row):

        return 1.0

    def sprinkle_sne(self, sne_density=1.0, rand_state=None):

        if rand_state is None and type(rand_state) != int:
            rand_state = np.random.RandomState(49)
        elif rand_state is None and type(rand_state) == int:
            rand_state = np.random.RandomState(rand_state)

        # Get rid of galaxies with no host type
        # (we set type to None when they are too small)
        sne_gals = self.gal_cat.iloc[np.where(self.gal_cat['gal_type'] ==
                                              self.gal_cat['gal_type'])]
        sne_ids = sne_gals['galaxy_id'].values
        if self.avoid_gal_ids is not None:
            sne_gals = sne_gals.iloc[[x for x in np.arange(len(sne_ids))
                                      if sne_ids[x] not in self.avoid_gal_ids]]
        sne_match_idx = self.match_sne(sne_gals['redshift'].values,
                                       sne_gals['gal_type'].values)

        return
