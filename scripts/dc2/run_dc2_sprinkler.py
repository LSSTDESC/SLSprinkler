import argparse
import os
import sys
sys.path.append('../..')
from sprinkler import OM10Reader, GoldsteinSNeCatReader, DC2Reader, DC2Sprinkler

def main(input_dir, agn_input, sne_input, dc2_input,
         output_dir, agn_output, sne_output, lens_output, host_output):

    agn_cat = os.path.join(input_dir, agn_input)
    sne_cat = os.path.join(input_dir, sne_input)
    dc2_cat = os.path.join(input_dir, dc2_input)

    dc2_sprinkler = DC2Sprinkler(input_dir, agn_cat, OM10Reader,
                                 sne_cat, GoldsteinSNeCatReader,
                                 dc2_cat, DC2Reader)

    agn_hosts, agn_systems, sne_hosts, sne_systems = \
        dc2_sprinkler.generate_matched_catalogs()

    print('Writing AGN')
    agn_output_path = os.path.join(output_dir, agn_output)
    dc2_sprinkler.output_lensed_agn_truth(agn_hosts, agn_systems,
                                          agn_output_path, id_offset=0)
    print('Writing SNe')
    sne_output_path = os.path.join(output_dir, sne_output)
    dc2_sprinkler.output_lensed_sne_truth(sne_hosts, sne_systems,
                                          sne_output_path, id_offset=2000)
    print('Writing Lenses')
    lens_output_path = os.path.join(output_dir, lens_output)
    dc2_sprinkler.output_lens_galaxy_truth(agn_hosts, agn_systems, sne_hosts,
                                           sne_systems, lens_output_path)
    print('Writing Hosts')
    host_output_path = os.path.join(output_dir, host_output)
    dc2_sprinkler.output_host_galaxy_truth(agn_hosts, agn_systems, sne_hosts,
                                           sne_systems, host_output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run DC2 Sprinkler and generate truth catalogs.'
    )
    parser.add_argument('--input_dir', type=str, help='Input Data Directory')
    parser.add_argument('--output_dir', type=str, help='Output Data Directory')
    args = parser.parse_args()

    agn_input = 'twinkles_lenses_cosmoDC2_v1.1.4.fits'
    sne_input = 'glsne_cosmoDC2_v1.1.4.h5'
    dc2_input = 'full_ddf.pkl'

    agn_output = 'agn_truth.db'
    sne_output = 'sne_truth.db'
    lens_output = 'lens_truth.db'
    host_output = 'host_truth.db'

    main(args.input_dir, agn_input, sne_input, dc2_input, args.output_dir,
         agn_output, sne_output, lens_output, host_output)