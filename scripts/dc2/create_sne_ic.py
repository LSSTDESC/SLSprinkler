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


class lensedSneCat(instCatUtils):

    def __init__(self, truth_cat, out_dir, cat_file_name,
                 sed_folder_name, write_sn_sed=True):

        self.truth_cat = truth_cat
        self.sn_obj = SNObject(0., 0.)
        self.imSimBand = Bandpass()
        self.imSimBand.imsimBandpass()
        self.sed_folder_name = sed_folder_name
        self.out_dir = out_dir
        self.sed_dir = os.path.join(out_dir, sed_folder_name)
        self.write_sn_sed = write_sn_sed

        if not os.path.exists(self.sed_dir):
            os.mkdir(self.sed_dir)

    def calc_sne_mags(self, obs_mjd, obs_filter):

        wavelen_max = 1800.
        wavelen_min = 30.
        wavelen_step = 0.1

        sn_magnorm_list = []
        sn_sed_names = []
        add_to_cat_list = []

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

            sn_sed_obj = current_sn_obj.SNObjectSourceSED(time=sed_mjd,
                                                          wavelen=np.arange(wavelen_min,
                                                                            wavelen_max,
                                                                            wavelen_step))
            flux_500 = sn_sed_obj.flambda[np.where(sn_sed_obj.wavelen >= 499.99)][0]

            if flux_500 > 0.:
                sn_magnorm = sn_sed_obj.calcMag(bandpass=self.imSimBand)
                sn_name = None
                if self.write_sn_sed:
                    sn_name = '%s/specFileGLSN_%s_%s_%.4f.txt' % (self.sed_folder_name,
                                                                  self.truth_cat['dc2_sys_id'].iloc[idx],
                                                                  self.truth_cat['image_number'].iloc[idx],
                                                                  obs_mjd)
                    sed_filename = '%s/%s' % (self.out_dir, sn_name)
                    sn_sed_obj.writeSED(sed_filename)
                    with open(sed_filename, 'rb') as f_in, gzip.open(str(sed_filename + '.gz'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(sed_filename)

                sn_magnorm_list.append(sn_magnorm)
                sn_sed_names.append(sn_name + '.gz')
                add_to_cat_list.append(idx)

        return add_to_cat_list, np.array(sn_magnorm_list), sn_sed_names

    def output_instance_catalog(self, add_to_cat_idx, sne_magnorms, sne_sed_names,
                                obs_md, filename):

        full_cat_name = os.path.join(self.out_dir, filename)

        lensed_mags = sne_magnorms - \
            2.5*np.log10(np.abs(self.truth_cat['magnification'].iloc[add_to_cat_idx].values))

        phosim_coords = self.get_phosim_coords(np.radians(self.truth_cat['ra'].values),
                                               np.radians(self.truth_cat['dec'].values),
                                               obs_md)
        phosim_ra, phosim_dec = np.degrees(phosim_coords)

        with open(full_cat_name, 'w') as f:
            for truth_cat_idx, output_idx in zip(add_to_cat_idx, np.arange(len(add_to_cat_idx))):

                f.write('object %s %f %f %f %s %f 0 0 0 0 0 point none CCM %f %f\n' \
                    % ('%s_%s' % (self.truth_cat['dc2_sys_id'].iloc[truth_cat_idx],
                                  self.truth_cat['image_number'].iloc[truth_cat_idx]),
                       phosim_ra[truth_cat_idx],
                       phosim_dec[truth_cat_idx],
                       lensed_mags[output_idx],
                       sne_sed_names[output_idx],
                       self.truth_cat['redshift'].iloc[truth_cat_idx],
                       self.truth_cat['av_mw'].iloc[truth_cat_idx],
                       self.truth_cat['rv_mw'].iloc[truth_cat_idx]))

        return

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
    obs_time = obs_md.mjd.TAI
    obs_filter = obs_md.bandpass
    print('Writing Instance Catalog for Visit: %i at MJD: %f in Bandpass: %s' % (args.obs_id,
                                                                                 obs_time,
                                                                                 obs_filter))
    add_to_cat_idx, sne_magnorms, sne_sed_names = lensed_sne_ic.calc_sne_mags(obs_time, obs_filter)
    lensed_sne_ic.output_instance_catalog(add_to_cat_idx, sne_magnorms,
                                          sne_sed_names, obs_md, args.cat_file_name)
