import os
import sqlite3
import numpy as np
import scipy.special as ss
import pylab as pl
import pandas as pd
from astropy.io import fits
import om10_lensing_equations as ole

__all__ = ['LensedHostGenerator', 'generate_lensed_host',
           'lensed_sersic_2d', 'random_location']


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


def random_location(Reff_src, qs, phs, ns, rng=None):
    """Sample a random (x, y) location from the surface brightness
    profile of the galaxy. The input parameters are Sersic parameters for the host galaxy.
    Parameters:
    -----------
    Reff_src: float
        the effective radius in arcseconds, the radius within which half of the light is contained
    qs: float
        axis ratio of the source, b/a
    phs: float
        position angle of the galaxy in degrees
    ns: int
        Sersic index
    rng: numpy.random.RandomState [None]
        RandomState object to use for generating random draws from [0, 1).
        If None, then create a RandomState with default seeding.

    Returns:
    -----------
    dx: horizontal coordinate of random location (pixel coordinates)
    dy: vertical coordinate of random location (pixel coordinates)
    """
    if rng is None:
        rng = np.random.RandomState()

    phs_rad = np.deg2rad(phs-90)

    bn = ss.gammaincinv(2. * ns, 0.5)
    z = rng.random_sample()
    x = ss.gammaincinv(2. * ns, z)
    R = (x / bn)**ns * Reff_src
    theta = rng.random_sample() * 2 * np.pi

    xp, yp = R * np.cos(theta), R * np.sin(theta)
    xt = xp * np.sqrt(qs)
    yt = yp / np.sqrt(qs)
    dx, dy = np.linalg.solve([[np.cos(phs_rad), np.sin(phs_rad)],
                             [-np.sin(phs_rad), np.cos(phs_rad)]],
                             [xt, yt])
    return dx, dy


def check_random_locations():
    """Defines a random location to compare to"""

    npoints = 100000
    Reff_disk = 0.2
    qs_disk = 0.3
    phs_disk = 8.
    ns_disk = 1.0

    x_d = np.zeros(npoints)
    y_d = np.zeros(npoints)

    for i in range(npoints):
        x_d[i], y_d[i] = random_location(Reff_disk, qs_disk, phs_disk, ns_disk)


    bsz = 5.0
    nnn = 1000  # number of pixels per side
    dsx = bsz/nnn
    xi1, xi2 = ole.make_r_coor(nnn, dsx)

    src_disk = ole.sersic_2d(xi1,xi2,0.0,0.0,Reff_disk,qs_disk,phs_disk,ns_disk)
    src_disk_norm = src_disk/(np.sum(src_disk)*dsx*dsx)

    src_disk_px = np.sum(src_disk, axis=1)
    src_disk_norm_px = src_disk_px/(np.sum(src_disk_px)*dsx)

    src_disk_py = np.sum(src_disk, axis=0)
    src_disk_norm_py = src_disk_py/(np.sum(src_disk_py)*dsx)

    xsteps = xi1[:,0]
    #---------------------------------------------------------------

    from matplotlib.ticker import NullFormatter

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    pl.figure(1, figsize=(8, 8))

    axScatter = pl.axes(rect_scatter)
    axHistx = pl.axes(rect_histx)
    axHisty = pl.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x_d, y_d)
    axScatter.contour(xi1, xi2, src_disk, colors=['k',])

    # now determine nice limits by hand:
    binwidth = 0.02
    xymax = max(np.max(np.abs(x_d)), np.max(np.abs(y_d)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x_d, bins=bins, density=1)
    axHistx.plot(xsteps, src_disk_norm_px, 'k-')

    axHisty.hist(y_d, bins=bins, density=1,orientation='horizontal')
    axHisty.plot(src_disk_norm_py, xsteps, 'k-')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    return 0


class LensedHostGenerator:
    """Class to generate lensed hosts."""
    def __init__(self, host_truth_file, lens_truth_file, obj_type, outdir,
                 pixel_size=0.04, num_pix=250, rng=None):
        with sqlite3.connect(host_truth_file) as conn:
            host_df = pd.read_sql(f'select * from {obj_type}_hosts', conn) \
                        .query('image_number==0')
        with sqlite3.connect(lens_truth_file) as conn:
            lens_df = pd.read_sql(f'select * from {obj_type}_lens', conn)
        self.df = pd.merge(host_df, lens_df, on='lens_cat_sys_id', how='inner')
        self.obj_type = obj_type
        self.outdir = outdir
        self.pixel_size = pixel_size
        self.xi1, self.xi2 = ole.make_r_coor(num_pix, pixel_size)
        self.rng = rng

    def create(self, index):
        """Generate the lensed host for the object pointed at by `index`"""
        lens_params, bulge_params, disk_params = self._extract_params(index)
        generate_lensed_host(self.xi1, self.xi2, lens_params, bulge_params,
                             disk_params, self.pixel_size, self.outdir,
                             self.obj_type)

    def _extract_params(self, index):
        row = self.df.iloc[index]

        if not (np.isfinite(row['x_src']) and np.isfinite(row['y_src'])):
            raise RuntimeError('x_src or y_src is not finite for '
                               f'lens id {row["lens_cat_sys_id"]}')

        dc2_sys_id_tokens = row['dc2_sys_id_x'].split('_')
        UID_lens = '_'.join((dc2_sys_id_tokens[0], 'host', dc2_sys_id_tokens[1],
                             str(row['image_number'])))
        twinkles_ID = UID_lens
        lens_cat = {'xl1': 0,
                    'xl2': 0,
                    'ql': 1.0 - row['ellip_lens'],
                    'vd': row['vel_disp_lenscat'],
                    'phl': row['phie_lens'],
                    'gamma': row['gamma_lenscat'],
                    'phg': row['phig_lenscat'],
                    'zl': row['redshift_y'],
                    'ximg': row['x_img'],
                    'yimg': row['y_img'],
                    'twinklesid': twinkles_ID,
                    'lensid': row['lens_cat_sys_id'],
                    'index': index,
                    'UID_lens': UID_lens,
                    'Ra_lens': row['ra_lens_x'],
                    'Dec_lens': row['dec_lens_x'],
                    'cat_id': row['unique_id_x']}

        if self.rng is not None:
            offsets = self._random_offsets(row, self.rng)
        else:
            offsets = None
        srcsP_bulge = self._extract_source_params(row, 'bulge', offsets=offsets)
        srcsP_disk = self._extract_source_params(row, 'disk', offsets=offsets)

        return lens_cat, srcsP_bulge, srcsP_disk

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _random_offsets(row, rng):
        # The original code used the disk component for computing
        # the offsets for both the bulge and disk components.
        Reff = np.sqrt(row['minor_axis_disk']*row['major_axis_disk'])
        qs = row['minor_axis_disk']/row['major_axis_disk']
        # The original code had this unexpected ordering of dys2 and dys1:
        dys2, dys1 = random_location(Reff, qs, row['position_angle_x'],
                                     row['sindex_disk'], rng)
        # Return the offsets in the expected order:
        return dys1, dys2

    @staticmethod
    def _extract_source_params(row, component, offsets=None):
        Reff = np.sqrt(row[f'minor_axis_{component}']
                       *row[f'major_axis_{component}'])
        qs = row[f'minor_axis_{component}']/row[f'major_axis_{component}']
        ys1 = row['x_src']
        ys2 = row['y_src']
        if offsets is not None:
            ys1 -= offsets[0]
            ys2 -= offsets[1]
        return {'ys1': ys1,
                'ys2': ys2,
                'mag_src_u': row[f'magnorm_{component}_u'],
                'mag_src_g': row[f'magnorm_{component}_g'],
                'mag_src_r': row[f'magnorm_{component}_r'],
                'mag_src_i': row[f'magnorm_{component}_i'],
                'mag_src_z': row[f'magnorm_{component}_z'],
                'mag_src_y': row[f'magnorm_{component}_y'],
                'Reff_src': Reff,
                'qs': qs,
                'phs': row['position_angle_x'],
                'ns': row[f'sindex_{component}'],
                'zs': row['redshift_x'],
                'sed_src': row[f'sed_{component}_host'],
                'components': component}
