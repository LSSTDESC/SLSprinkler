import os
import numpy as np
import pandas as pd
import argparse
from sqlalchemy import create_engine
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from desc.sims.GCRCatSimInterface import get_obs_md
from lsst.sims.utils import angularSeparation

__all__ = ['hostImage']

class hostImage(object):
    """Takes FITS stamps and includes them in instance catalogs
    hostImage takes the following arguments:
    ra_center: Right ascension of the center of the field (in degrees)
    dec_center: Declination of the center of the field (in degrees)
    fov: Field-of-view angular radius (in degrees).  2 degrees will cover
        the LSST focal plane."""

    def __init__(self, ra_center, dec_center, fov, bandpass):

        self.ra = ra_center
        self.dec = dec_center
        self.radius = fov
        self.bandpass = bandpass

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

        fits_info = fits_file_name.split('_')
        lens_id = np.int(fits_info[0])
        sys_magNorm_list = np.array(fits_info[1:-1], dtype=np.float)
        sys_magNorm = sys_magNorm_list[self.bandpass_lookup[self.bandpass]]
        gal_type = fits_info[-1].split('.')[0]

        # galaxy_id = np.right_shift(lens_id, 10)
        # galaxy_id *= 10000
        # galaxy_id += 4*df_line['twinkles_system']
        # sys_id = np.left_shift(galaxy_id, 10)
        # if gal_type == 'bulge':
        #     sys_id += 97
        # elif gal_type == 'disk':
        #     sys_id += 107
        # else:
        #     raise ValueError('Invalid Galaxy Component Type in filename')
        sys_id = lens_id

        cat_str = 'object %i %f %f %f %s %f 0 0 0 0 0 %s 0.01 0 CCM %f %f CCM %f %f\n' % (sys_id,
                                                                      df_line['ra_lens'],
                                                                      df_line['dec_lens'],
                                                                      sys_magNorm,
                                                                      df_line['sed_%s_host' % gal_type].decode('utf-8'),
                                                                      df_line['redshift'],
                                                                      str(image_dir)+'/'+fits_file_name,
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
        image_ids = np.array([image_name.split('_')[0] for image_name in image_list], dtype=np.int)
        
        for sys_ra, sys_dec in zip(host_df['ra_lens'], host_df['dec_lens']):
            ang_sep_list.append(angularSeparation(sys_ra, sys_dec, self.ra, self.dec))

        ang_sep = np.array(ang_sep_list)
        keep_idx = np.where(ang_sep < self.radius)
        host_image_df = host_df.iloc[keep_idx]

        unique_id_list = []

        if append:
            write_status = 'a'
        else:
            write_status = 'w'

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

    agn_host_image = hostImage(obs_md.pointingRA, obs_md.pointingDec, args.fov, obs_filter)
    sne_host_image = hostImage(obs_md.pointingRA, obs_md.pointingDec, args.fov, obs_filter)

    agn_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'agn_lensed_disks'), agn_host_truth_cat,
                                  args.file_out, append=False)
    agn_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'agn_lensed_bulges'), agn_host_truth_cat,
                                  args.file_out, append=True)
    sne_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'sne_lensed_disks'), sne_host_truth_cat,
                                  args.file_out, append=True)
    sne_host_image.write_host_cat(os.path.join(args.fits_stamp_dir, 'sne_lensed_bulges'), sne_host_truth_cat,
                                  args.file_out, append=True)
    
