# -*- coding: utf-8 -*-
"""Utility functions for reading and writing tables

"""

import os
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from astropy.io import fits
import pandas as pd

__all__ = ['to_csv', 'export_db']

def to_csv(truth_db_path, dest_dir='.', table_suffix=''):
    """Dumps sqlite3 files of the truth tables as csv files

    Parameters
    ----------
    truth_db_path : str
        Path to the truth tables Bryce made, either of the lens or the host

    """
    db = sqlite3.connect(truth_db_path)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(os.path.join(dest_dir, table_name + '{:s}.csv'.format(table_suffix)), index=False)
    cursor.close()
    db.close()

def export_db(dataframe, out_dir, out_fname, table_name, overwrite=False):
    """Export a DB from a Pandas DataFrame

    Parameters
    ----------
    dataframe : Pandas.DataFrame object
    out_fname : str
    table_name : str
    overwrite_existing : bool
    
    """
    out_path = os.path.join(out_dir, out_fname)
    if overwrite is True and os.path.exists(out_path):
        os.remove(out_path)
    engine = create_engine('sqlite:///{:s}'.format(out_path), echo=False)
    dataframe.to_sql(table_name, con=engine, index=False)
    engine.dispose()
    return None

def boundary_max(data):
    """Get the maximum pixel value along the four boundaries of the given image

    """
    ny, nx = data.shape
    boundary = np.concatenate((data[:, 0], data[:, -1], data[0, :],
                               data[-1, :]))
    return np.max(boundary)

def write_fits_stamp(data, magnorms, lens_id, galaxy_type, pixel_scale, outfile, overwrite=True):
    """Write the given image as a fits stamp with relevant metadata

    Parameters
    ----------
    data : np.array
        the image to export
    magnorms : dict
        the normalizing magnitude with ugrizy keys
    lens_id : int
        the dc2_sys_id for this lensing system
    galaxy_type : str
        the galaxy component type ('bulge' or 'disk')
    pixel_scale : float
    outfile : output file path
    overwrite : bool

    """

    boundary_ratio = boundary_max(data)/np.max(data)
    if boundary_ratio > 1e-2:
        print(f'(boundary max/data max) = {boundary_ratio:.2e} '
              f'for {galaxy_type} {lens_id}')
    for magnorm in magnorms.values():
        if not np.isfinite(magnorm):
            raise RuntimeError(f'non-finite magnorm for {lens_id}')
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    output = fits.HDUList(fits.PrimaryHDU())
    output[0].data = data
    output[0].header.set('LENS_ID', lens_id, 'Lens system ID')
    output[0].header.set('GALTYPE', galaxy_type, 'Galaxy component type')
    for band, magnorm in magnorms.items():
        output[0].header.set(f'MAGNORM{band.upper()}', magnorm,
                             f'magnorm for {band}-band')
    output[0].header.set('PIXSCALE', pixel_scale, 'pixel scale in arcseconds')
    output.writeto(outfile, overwrite=overwrite)

