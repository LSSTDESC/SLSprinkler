#!/usr/bin/env python
"""Script to generate postage-stamp fits files of lensed AGN hosts.

Example
-------
To use the default settings, run this script from the root of the repo with the required `object_type` argument::
    
    $ python generate_lensed_host_agn.py agn

The fits files will be generated in the `outputs` folder.

"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import lensing_utils
import io_utils

# Have numpy raise exceptions for operations that would produce nan or inf.
np.seterr(invalid='raise', divide='raise', over='raise')

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser(description="Script to generate postage-stamp fits files of lensed AGN hosts")
    parser.add_argument("object_type", type=str,
                        help="Type of object the source galaxy hosts ('agn' or 'sne')")
    parser.add_argument("--datadir", type=str, default='truth_tables',
                    help='Location of directory containing truth tables')
    parser.add_argument("--outdir", type=str, default='outputs',
                        help='Output location for FITS stamps')
    parser.add_argument("--pixel_size", type=float, default=0.04,
                        help='Pixel size in arcseconds')
    parser.add_argument("--num_pix", type=int, default=250,
                        help='Number of pixels in x- or y-direction')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_dir = args.datadir
    output_dir = args.outdir
    object_type = args.object_type
    # Convert to dataframes for easy manipulation
    lens_df = pd.read_sql('%s_lens' % object_type,
                          os.path.join('sqlite:///', input_dir, 'lens_truth.db'), index_col=0)
    src_light_df = pd.read_sql('%s_hosts' % object_type,
                               os.path.join('sqlite:///', input_dir, 'host_truth.db'), index_col=0)
    # Instantiate tool for imaging our hosts
    lensed_host_imager = lensing_utils.LensedHostImager(args.pixel_size, args.num_pix)
    sys_ids = lens_df['lens_cat_sys_id'].unique()
    lens_id = dict()
    for dc2_sys_id, sys_id in zip(lens_df['dc2_sys_id'], lens_df['lens_cat_sys_id']):
        tokens = dc2_sys_id.split('_')
        lens_id[sys_id] = '_'.join((tokens[0], 'host', tokens[1], '0'))
    progress = tqdm(total=len(sys_ids))
    for i, sys_id in enumerate(sys_ids):
        lens_info = lens_df.loc[lens_df['lens_cat_sys_id']==sys_id].squeeze()
        src_light_info = src_light_df.loc[src_light_df['lens_cat_sys_id']==sys_id].iloc[0].squeeze() # arbitarily take the first lensed image, since the source properties are the same between the images
        # Get images and some metadata
        z_lens = lens_info['redshift']
        z_src = src_light_info['redshift']
        bulge_img, bulge_features = lensed_host_imager.get_image(lens_info, src_light_info, z_lens, z_src, 'bulge')
        disk_img, disk_features = lensed_host_imager.get_image(lens_info, src_light_info, z_lens, z_src, 'disk')
        # Export images with metadata
        bulge_out_path = os.path.join(output_dir, f'{object_type}_lensed_bulges', f"{lens_id[sys_id]}_bulge.fits")
        disk_out_path = os.path.join(output_dir, f'{object_type}_lensed_disks', f"{lens_id[sys_id]}_disk.fits")
        io_utils.write_fits_stamp(bulge_img, bulge_features['magnorms'], lens_id[sys_id], 'bulge', args.pixel_size, bulge_out_path)
        io_utils.write_fits_stamp(disk_img, disk_features['magnorms'], lens_id[sys_id], 'disk', args.pixel_size, disk_out_path)
        progress.update(1)
    progress.close()

if __name__ == '__main__':
    main()
