import os
import numpy as np
import pandas as pd
import argparse
from astropy.io import fits
from sqlalchemy import create_engine
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from desc.sims.GCRCatSimInterface import get_obs_md
from lsst.sims.utils import angularSeparation
from dc2_utils import instCatUtils

__all__ = ['hostImage']

class hostImage(instCatUtils):
    """Takes FITS stamps and includes them in instance catalogs
    hostImage takes the following arguments:
    ra_center: Right ascension of the center of the field (in degrees)
    dec_center: Declination of the center of the field (in degrees)
    fov: Field-of-view angular radius (in degrees).  2 degrees will cover
        the LSST focal plane."""

    def __init__(self, obs_md, fov):

        self.ra = obs_md.pointingRA
        self.dec = obs_md.pointingDec
        self.radius = fov
        self.bandpass = obs_md.bandpass
        self.obs_md = obs_md

        self.bandpass_lookup = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}


    def format_catalog(self, df_line, fits_file_name, image_dir):
        """
        Formats the output instance catalog to include entries for the FITS
        stamps produced by generate_lensed_host_***.py

		Parameters:
		-----------
		df_line: string
			pandas data frame line
		fits_file_name: string
			the filename of the FITS stamp
		image_dir: string
			the location of the FITS stamps
		Returns:
		-----------
		cat_str: a string containing the line of parameters to go into an instance file
        """
        with fits.open(os.path.join(image_dir, fits_file_name)) as hdus:
            lens_id = hdus[0].header['LENS_ID']
            sys_magNorm_list = [hdus[0].header[f'MAGNORM{_}'] for _ in 'UGRIZY']
            sys_magNorm = sys_magNorm_list[self.bandpass_lookup[self.bandpass]]
            gal_type = hdus[0].header['GALTYPE'].strip()

        if gal_type == 'bulge':
            sys_id = lens_id + '_b'
        elif gal_type == 'disk':
            sys_id = lens_id + '_d'

        sed_file = df_line['sed_%s_host' % gal_type]
        if isinstance(sed_file, bytes):
            sed_file = sed_file.decode('utf-8')
        else:
            sed_file = sed_file.lstrip('b').strip("'")
        cat_str = 'object %s %f %f %f %s %f 0 0 0 0 0 %s 0.01 0 CCM %f %f CCM %f %f\n'\
                  % (sys_id,
                     df_line['ra_lens'],
                     df_line['dec_lens'],
                     sys_magNorm,
                     sed_file,
                     df_line['redshift'],
                     os.path.basename(str(image_dir))+'/'+fits_file_name,
                     df_line['av_internal_%s' % gal_type],
                     df_line['rv_internal_%s' % gal_type],
                     df_line['av_mw'],
                     df_line['rv_mw'])

        return cat_str

    def write_host_cat(self, image_dir, host_df, output_cat, append=False):
        """Adds entries for each lensed host FITS stamp to output instance catalog
        Parameters:
        -----------
        image_dir: string
            the location of the FITS stamps
        host_df: pandas dataframe
            the agn/sne host truth catalog in pandas dataframe format
        output_cat: string
            the location of the output instance catalogs """

        ang_sep_list = []
        image_list = os.listdir(image_dir)
        image_ids = np.array(['_'.join(image_name.split('_')[:4])
                              for image_name in image_list], dtype=str)

        for sys_ra, sys_dec in zip(host_df['ra_lens'], host_df['dec_lens']):
            ang_sep_list.append(angularSeparation(sys_ra, sys_dec, self.ra, self.dec))

        ang_sep = np.array(ang_sep_list)
        keep_idx = np.where(ang_sep < self.radius)
        host_image_df = host_df.iloc[keep_idx].reset_index(drop=True)

        unique_id_list = []

        if append:
            write_status = 'a'
        else:
            write_status = 'w'

        phosim_coords = self.get_phosim_coords(np.radians(host_image_df['ra_lens'].values),
                                               np.radians(host_image_df['dec_lens'].values),
                                               self.obs_md)
        phosim_ra, phosim_dec = np.degrees(phosim_coords)

        host_image_df['ra_lens'] = phosim_ra
        host_image_df['dec_lens'] = phosim_dec

        with open(output_cat, write_status) as f:
            for df_row_num in range(len(host_image_df)):
                df_line = host_image_df.iloc[df_row_num]
                line_uID = df_line['unique_id']
                if line_uID in unique_id_list:
                    continue
                line_idx = np.where(image_ids == line_uID)[0]
                if len(line_idx) > 1:
                    print(line_idx, line_uID)
                    raise ValueError('Multiple images have same unique lens id')
                if len(line_idx) > 0:
                    line_name = image_list[line_idx[0]]
                    f.write(self.format_catalog(df_line, line_name, image_dir))
                    unique_id_list.append(line_uID)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                'Lensed Host Instance Catalog Generator')
    parser.add_argument('--obs_db', type=str, help='path to the Opsim db')
    parser.add_argument('--obs_id', type=int, default=None,
                        help='obsHistID to generate InstanceCatalog for')
    parser.add_argument('--fov', type=float, help='size of field of view')
    parser.add_argument('--host_truth_cat', type=str,
                        help='path to lensed host truth catalog')
    parser.add_argument('--fits_stamp_dir', type=str,
                        help='directory with the lensed host stamps')
    parser.add_argument('--file_out', type=str,
                        help='filename of instance catalog written')

    args = parser.parse_args()

    host_truth_db = create_engine('sqlite:///%s' % args.host_truth_cat, echo=False)
    agn_host_truth_cat = pd.read_sql_table('agn_hosts', host_truth_db)
    sne_host_truth_cat = pd.read_sql_table('sne_hosts', host_truth_db)

    obs_gen = ObservationMetaDataGenerator(database=args.obs_db,
                                           driver='sqlite')
    obs_md = get_obs_md(obs_gen, args.obs_id, 2, dither=True)
    obs_time = obs_md.mjd.TAI
    obs_filter = obs_md.bandpass
    print('Writing Instance Catalog for Visit: %i at MJD: %f in Bandpass: %s' % (args.obs_id,
                                                                                 obs_time,
                                                                                 obs_filter))

    agn_host_image = hostImage(obs_md, args.fov)
    sne_host_image = hostImage(obs_md, args.fov)

    agn_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'agn_lensed_disks'), agn_host_truth_cat,
                                  args.file_out, append=False)
    agn_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'agn_lensed_bulges'), agn_host_truth_cat,
                                  args.file_out, append=True)
    sne_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'sne_lensed_disks'), sne_host_truth_cat,
                                  args.file_out, append=True)
    sne_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'sne_lensed_bulges'), sne_host_truth_cat,
                                  args.file_out, append=True)
