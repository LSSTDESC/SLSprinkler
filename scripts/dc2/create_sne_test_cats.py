import os
import pandas as pd
import numpy as np
import gzip
import shutil
import copy
import json
import argparse
from sqlalchemy import create_engine
from lsst.utils import getPackageDir
from lsst.sims.photUtils import Bandpass
from lsst.sims.catUtils.supernovae import SNObject
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from desc.sims.GCRCatSimInterface import get_obs_md
from dc2_utils import instCatUtils
from create_sne_ic import lensedSneCat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                'Lensed SNe Instance Catalog Generator')
    parser.add_argument('--obs_db', type=str, help='path to the Opsim db')
    parser.add_argument('--obs_id', type=int, default=None,
                        help='obsHistID to generate InstanceCatalog for')
    parser.add_argument('--sne_truth_cat', type=str,
                        help='path to lensed AGN truth catalog')
    parser.add_argument('--output_dir', type=str,
                        help='output directory for catalog and sed folder')
    parser.add_argument('--cat_file_name', type=str,
                        help='filename of instance catalog written')
    parser.add_argument('--sed_folder', type=str,
                        help='directory to put SNe SEDs. Will appear in output_dir.')

    args = parser.parse_args()

    obs_gen = ObservationMetaDataGenerator(database=args.obs_db,
                                           driver='sqlite')

    sne_truth_db = create_engine('sqlite:///%s' % args.sne_truth_cat, echo=False)
    sne_truth_cat = pd.read_sql_table('lensed_sne', sne_truth_db)
    lensed_sne_ic = lensedSneCat(sne_truth_cat, args.output_dir,
                                 args.cat_file_name, args.sed_folder)

    obs_md = get_obs_md(obs_gen, args.obs_id, 2, dither=True)
    print(obs_md.mjd.TAI)
    for obs_time in np.arange(obs_md.mjd.TAI, obs_md.mjd.TAI + 35.1, 0.25):
        obs_filter = obs_md.bandpass
        print('Writing Instance Catalog for Visit: %i at MJD: %f in Bandpass: %s' % (args.obs_id,
                                                                                     obs_time,
                                                                                     obs_filter))
        add_to_cat_idx, sne_magnorms, sne_sed_names = lensed_sne_ic.calc_sne_mags(obs_time, obs_filter)
        lensed_sne_ic.output_instance_catalog(add_to_cat_idx, sne_magnorms,
                                            sne_sed_names, obs_md, str('test_cat_%.4f' % obs_time))
