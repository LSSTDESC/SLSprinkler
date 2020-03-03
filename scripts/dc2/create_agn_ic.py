from builtins import range
from builtins import object
import argparse
import linecache
import math
import os
import gzip
import numbers
import multiprocessing
import json as json
import pandas as pd
import numpy as np
from lsst.sims.catalogs.decorators import register_method, compound
from sqlalchemy import create_engine
from lsst.sims.photUtils import Sed, BandpassDict
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from desc.sims.GCRCatSimInterface import get_obs_md
from dc2_utils import ExtraGalacticVariabilityModels


class lensedAgnCat():

    def __init__(self, truth_cat):

        self.truth_cat = truth_cat
        self.filter_num_dict = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'y':5}

        return

    def calc_agn_dmags(self, obs_mjd, obs_filter):

        agn_param_colnames = ['seed', 'agn_sf_u', 'agn_sf_g', 'agn_sf_r', 'agn_sf_i',
                              'agn_sf_z', 'agn_sf_y', 'agn_tau_u', 'agn_tau_g',
                              'agn_tau_r', 'agn_tau_i', 'agn_tau_z', 'agn_tau_y']
        agn_params = {}
        for colname in agn_param_colnames:
            agn_params[colname] = self.truth_cat[colname]

        # Include time delay.
        obs_mjd_delay = obs_mjd - self.truth_cat['t_delay']

        agn_simulator = ExtraGalacticVariabilityModels()
        agn_simulator._agn_threads = 1
        agn_simulator.filters_to_simulate = obs_filter
        d_mag = agn_simulator.applyAgn([np.arange(len(self.truth_cat), dtype=int)],
                                       agn_params, obs_mjd_delay,
                                       redshift=self.truth_cat['redshift'])

        return d_mag[self.filter_num_dict[obs_filter]]

    def output_instance_catalog(self, d_mag, filename):

        lensed_mags = self.truth_cat['magnorm'] + d_mag - 2.5*np.log10(np.abs(self.truth_cat['magnification']))

        with open(filename, 'w') as f:
            for row_idx in range(len(self.truth_cat)):

                f.write('object %i %f %f %f agnSED/agn.spec.gz %f 0 0 0 0 0 point none CCM %f %f\n' \
                    % (self.truth_cat['unique_id'].iloc[row_idx],
                       self.truth_cat['ra'].iloc[row_idx],
                       self.truth_cat['dec'].iloc[row_idx],
                       lensed_mags[row_idx],
                       self.truth_cat['redshift'].iloc[row_idx],
                       self.truth_cat['av_mw'].iloc[row_idx],
                       self.truth_cat['rv_mw'].iloc[row_idx]))

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

    # obs_gen = ObservationMetaDataGenerator(database=args.obs_db,
    #                                        driver='sqlite')

    agn_truth_db = create_engine('sqlite:///%s' % args.agn_truth_cat, echo=False)
    agn_truth_cat = pd.read_sql_table('lensed_agn', agn_truth_db)
    lensed_agn_ic = lensedAgnCat(agn_truth_cat)

    for obsHistID in args.obs_ids:
        # obs_md = get_obs_md(obs_gen, obsHistID, 2, dither=True)
        # lensed_agn_ic.calc_agn_dmags(obs_md.mjd.TAI)
        obs_filter = 'g'
        d_mag = lensed_agn_ic.calc_agn_dmags(61000.0, obs_filter)
        lensed_agn_ic.output_instance_catalog(d_mag, 'test.out')
    