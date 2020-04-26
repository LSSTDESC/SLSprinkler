import os
import numpy as np
from astropy.io import fits
import om10_lensing_equations as ole

__all__ = ['generate_lensed_host', 'lensed_sersic_2d']


def boundary_max(data):
    ny, nx = data.shape
    boundary = np.concatenate((data[:, 0], data[:, -1], data[0, :],
                               data[-1, :]))
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


def lensed_sersic_2d(lens_pix, source_pix, source_cat):
    """
    Defines a magnitude of lensed host galaxy using 2d Sersic profile

    Parameters
    ----------
    lens_pix: (np.array, np.array)
        Arrays of xy pixel coordinates for the lens.
    source_pix: (np.array, np.array)
        Arrays of xy pixel coordinates for the bulge or disk.
    source_cat: dict-like
        Dictionary of source parameters.

    Returns
    -------
    (dict, np.array) dictionary of magnorms, lensed image
    """
    ysc1 = source_cat['ys1']      # x position of the source, arcseconds
    ysc2 = source_cat['ys2']      # y position of the source, arcseconds
    Reff = source_cat['Reff_src'] # Effective radius of the source, arcseconds
    qs = source_cat['qs']         # axis ratio of the source, b/a
    phs = source_cat['phs']       # orientation of the source, degree
    ndex = source_cat['ns']       # index of the source

    g_limage = ole.sersic_2d(*source_pix, ysc1, ysc2, Reff, qs, phs, ndex)
    g_source = ole.sersic_2d(*lens_pix, ysc1, ysc2, Reff, qs, phs, ndex)

    g_limage_sum = np.sum(g_limage)
    g_source_sum = np.sum(g_source)
    if g_limage_sum == 0 or g_source_sum == 0:
        raise RuntimeError('lensed image or soruce has zero-valued integral '
                           f'for lens id {source_cat["lensid"]}')
    dmag = -2.5*np.log10(g_limage_sum/g_source_sum)

    bands = 'ugrizy'
    mag_lensed = {band: source_cat[f'mag_src_{band}'] + dmag for band in bands}

    return mag_lensed, g_limage


def generate_lensed_host(xi1, xi2, lens_P, srcP_b, srcP_d, dsx, outdir,
                         object_type):
    """
    Does ray tracing of light from host galaxies using a non-singular
    isothermal ellipsoid profile, and writes out a FITS image files
    of the results of the ray tracing.

    Parameters
    ----------
    xi1: np.array
        Array of x-positions of lens image in pixel coordinates
    xi2: np.array
        Array of y-positions of lens image in pixel coordinates
    lens_P: dict
        Lens parameters (produced by create_cats_{object_type})
    srcP_b: dict
        Source bulge parameters (produced by create_cats_{object_type})
    srcP_d: dict
        Source disk parameters (produced by create_cats_{object_type})
    dsx: float
        Pixel scale in arcseconds
    """
    xlc1 = lens_P['xl1']         # x position of the lens, arcseconds
    xlc2 = lens_P['xl2']         # y position of the lens, arcseconds
    rlc = 0.0                    # core size of Non-singular Isothermal
                                 # Ellipsoid
    vd = lens_P['vd']            # velocity dispersion of the lens
    zl = lens_P['zl']            # redshift of the lens
    zs = srcP_b['zs']            # redshift of the source
    rle = ole.re_sv(vd, zl, zs)  # Einstein radius of lens, arcseconds.
    ql = lens_P['ql']            # axis ratio b/a
    le = ole.e2le(1.0 - ql)      # scale factor due to projection of ellpsoid
    phl = lens_P['phl']          # position angle of the lens, degree
    eshr = lens_P['gamma']       # external shear
    eang = lens_P['phg']         # position angle of external shear
    ekpa = 0.0                   # external convergence

    ai1, ai2 = ole.alphas_sie(xlc1, xlc2, phl, ql, rle, le, eshr, eang, ekpa,
                              xi1, xi2)

    yi1 = xi1 - ai1
    yi2 = xi2 - ai2

    lens_id = lens_P['UID_lens']

    magnorms, lensed_image_b = lensed_sersic_2d((xi1, xi2), (yi1, yi2), srcP_b)
    outfile = os.path.join(outdir, f'{object_type}_lensed_bulges',
                           f"{lens_id}_bulge.fits")
    write_fits_stamp(lensed_image_b, magnorms, lens_id, 'bulge', dsx, outfile)

    magnorms, lensed_image_d = lensed_sersic_2d((xi1, xi2), (yi1, yi2), srcP_d)
    outfile = os.path.join(outdir, f'{object_type}_lensed_disks',
                           f"{lens_id}_disk.fits")
    write_fits_stamp(lensed_image_d, magnorms, lens_id, 'disk', dsx, outfile)
