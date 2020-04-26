#!/usr/bin/env python
import os
import sys
import pylab as pl
import argparse
import numpy as np
from lensed_hosts_utils import LensedHostGenerator

# Have numpy raise exceptions for operations that would produce nan or inf.
np.seterr(invalid='raise', divide='raise', over='raise')

parser = argparse.ArgumentParser(description='Script to make SNe lensed hosts')
parser.add_argument("--datadir", type=str, default='truth_tables',
                    help='Location of input truth tables')
parser.add_argument("--outdir", type=str, default='outputs',
                    help='Output location for FITS stamps')
parser.add_argument("--pixel_size", type=float, default=0.01,
                    help='Pixel size in arcseconds')
parser.add_argument("--num_pix", type=int, default=1000,
                    help='Number of pixels in x- and y-directions')
parser.add_argument("--seed", type=int, default=42,
                    help='Seed for random draw of galaxy locations.')
args = parser.parse_args()
host_truth_file = os.path.join(args.datadir, 'host_truth.db')
lens_truth_file = os.path.join(args.datadir, 'lens_truth.db')
rng = np.random.RandomState(args.seed)

generator = LensedHostGenerator(host_truth_file, lens_truth_file, 'sne',
                                args.outdir, pixel_size=args.pixel_size,
                                num_pix=args.num_pix, rng=rng)

message_row = 0
message_freq = 50
num_rows = len(generator)
for i in range(num_rows):
    if i > 10:
        break
    if i >= message_row:
        print("working on system ", i, "of", num_rows)
        message_row += message_freq
    try:
        generator.create(i)
    except RuntimeError as eobj:
        print(eobj)
    sys.stdout.flush()
