import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine
from lsst.sims.photUtils import Sed, BandpassDict
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from desc.sims.GCRCatSimInterface import get_obs_md, ExtraGalacticVariabilityModels


class lensedAgnCat():

    def __init__(self, truth_cat):

        self.truth_cat = truth_cat

        return

    def calc_agn_dmags(self, obs_mjd):

        agn_param_colnames = ['seed', 'agn_sf_u', 'agn_sf_g', 'agn_sf_r', 'agn_sf_i',
                              'agn_sf_z', 'agn_sf_y', 'agn_tau_u', 'agn_tau_g',
                              'agn_tau_r', 'agn_tau_i', 'agn_tau_z', 'agn_tau_y']
        agn_params = {}
        for colname in agn_param_colnames:
            agn_params[colname] = self.truth_cat[colname]

        agn_simulator = ExtraGalacticVariabilityModels()
        agn_simulator._agn_threads = 1
        d_mag = agn_simulator.applyAgn([np.arange(len(self.truth_cat), dtype=int)],
                                       agn_params, obs_mjd,
                                       redshift=self.truth_cat['redshift'])

        return
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                'Lensed AGN Instance Catalog Generator')
    parser.add_argument('--obs_db', type=str, help='path to the Opsim db')
    parser.add_argument('--obs_ids', type=int, nargs='+', default=None,
                        help='obsHistID to generate InstanceCatalog for (a list)')
    parser.add_argument('--agn_truth_cat', type=str,
                        help='path to lensed AGN truth catalog')

    args = parser.parse_args()

    obs_gen = ObservationMetaDataGenerator(database=args.obs_db,
                                           driver='sqlite')

    agn_truth_db = create_engine('sqlite:///%s' % args.agn_truth_cat, echo=False)
    agn_truth_cat = pd.read_sql_table('lensed_agn', agn_truth_db)
    lensed_agn_ic = lensedAgnCat(agn_truth_cat)

    for obsHistID in args.obs_ids:
        obs_md = get_obs_md(obs_gen, obsHistID, 2, dither=True)
        lensed_agn_ic.calc_agn_dmags(obs_md.mjd.TAI+1.)
    