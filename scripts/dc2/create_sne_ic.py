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


class lensedSneCat():

    def __init__(self, truth_cat, sed_path, write_sn_sed=True):

        self.truth_cat = truth_cat
        self.sn_obj = SNObject(0., 0.)
        self.imSimBand = Bandpass()
        self.imSimBand.imsimBandpass()
        self.sed_path = sed_path
        self.write_sn_sed = write_sn_sed

    def calc_sne_mags(self, obs_mjd, obs_filter):

        wavelen_max = 1800.
        wavelen_min = 30.
        wavelen_step = 0.1

        for idx in range(len(self.truth_cat)):

            sn_param_dict = copy.deepcopy(self.sn_obj.SNstate)
            sn_param_dict['_ra'] = np.radians(self.truth_cat['ra'].iloc[idx])
            sn_param_dict['_dec'] = np.radians(self.truth_cat['dec'].iloc[idx])
            sn_param_dict['z'] = self.truth_cat['redshift'].iloc[idx]
            sn_param_dict['c'] = self.truth_cat['c'].iloc[idx]
            sn_param_dict['x0'] = self.truth_cat['x0'].iloc[idx]
            sn_param_dict['x1'] = self.truth_cat['x1'].iloc[idx]
            sn_param_dict['t0'] = self.truth_cat['t0'].iloc[idx]

            current_sn_obj = self.sn_obj.fromSNState(sn_param_dict)
            current_sn_obj.mwEBVfromMaps()
            sed_mjd = obs_mjd - self.truth_cat['t_delay'].iloc[idx]

            sn_sed_obj = current_sn_obj.SNObjectSED(time=sed_mjd,
                                                    wavelen=np.arange(wavelen_min, wavelen_max,
                                                                      wavelen_step))
            flux_500 = sn_sed_obj.flambda[np.where(sn_sed_obj.wavelen >= 499.99)][0]

            if flux_500 > 0.:
                add_to_cat = True
                sn_magnorm = current_sn_obj.catsimBandMag(self.imSimBand, sed_mjd)
                sn_name = None
                if self.write_sn_sed:
                    sn_name = 'specFileGLSN_%i_%i_%.4f.txt' % (self.truth_cat['lens_cat_sys_id'].iloc[idx],
                                                               self.truth_cat['image_number'].iloc[idx],
                                                               sed_mjd)
                    sed_filename = '%s/%s' % (self.sed_path, sn_name)
                    sn_sed_obj.writeSED(sed_filename)
                    with open(sed_filename, 'rb') as f_in, gzip.open(str(sed_filename + '.gz'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(sed_filename)

                print(sn_magnorm)

        return sn_magnorm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                'Lensed SNe Instance Catalog Generator')
    parser.add_argument('--obs_db', type=str, help='path to the Opsim db')
    parser.add_argument('--obs_id', type=int, default=None,
                        help='obsHistID to generate InstanceCatalog for')
    parser.add_argument('--sne_truth_cat', type=str,
                        help='path to lensed AGN truth catalog')
    parser.add_argument('--file_out', type=str,
                        help='filename of instance catalog written')
    parser.add_argument('--sed_out', type=str,
                        help='directory to put SNe SEDs')

    args = parser.parse_args()

    # obs_gen = ObservationMetaDataGenerator(database=args.obs_db,
    #                                        driver='sqlite')

    sne_truth_db = create_engine('sqlite:///%s' % args.sne_truth_cat, echo=False)
    sne_truth_cat = pd.read_sql_table('lensed_sne', sne_truth_db)
    lensed_sne_ic = lensedSneCat(sne_truth_cat, args.sed_out)

    # obs_md = get_obs_md(obs_gen, args.obs_id, 2, dither=True)
    obs_time = 60733.#obs_md.mjd.TAI
    obs_filter = 'g'#obs_md.bandpass
    print('Writing Instance Catalog for Visit: %i at MJD: %f in Bandpass: %s' % (args.obs_id,
                                                                                 obs_time,
                                                                                 obs_filter))
    d_mag = lensed_sne_ic.calc_sne_mags(obs_time, obs_filter)
    lensed_agn_ic.output_instance_catalog(d_mag, args.file_out)