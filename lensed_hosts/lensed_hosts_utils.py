import os
import numpy as np
from astropy.io import fits


def boundary_max(data):
    ny, nx = data.shape
    boundary = np.concatenate((data[:, 0], data[:, -1], data[0, :], data[-1, :]))
    return np.max(boundary)


def write_fits_stamp(data, magnorms, lens_id, galaxy_type, pixel_scale,
                     outfile, overwrite=True):
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
