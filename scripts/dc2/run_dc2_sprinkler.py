import argparse
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from sprinkler import OM10Reader, GoldsteinSNeCatReader, DC2Reader, DC2Sprinkler
from lsst.sims.catUtils.dust.EBV import EBVbase
from copy import deepcopy
import healpy
import h5py
from lsst.sims.photUtils import Sed, BandpassDict, getImsimFluxNorm, Bandpass
from lsst.utils import getPackageDir
from astropy.io import fits

def load_dc2_lenses(catalog_version, agn_db):

    # Load DC2 lens catalogs
    dc2_reader = DC2Reader(catalog_version)
    lens_cat_filters = ['ra < 53.755', 'ra > 52.495',
                        'dec < -27.55', 'dec > -28.65']
    lens_galaxy_df = dc2_reader.load_galaxy_catalog(lens_cat_filters)
    trim_lens_df = dc2_reader.trim_catalog(lens_galaxy_df)

    # Load DC2 AGN
    ddf_agn_query = 'ra < 53.755 and ra > 52.495 and dec < -27.55 and dec > -28.65'
    ddf_agn_df = dc2_reader.load_agn_catalog(agn_db, ddf_agn_query)

    # Remove any potential lens galaxies with AGN
    lens_id_list = list(trim_lens_df['galaxy_id'].values)
    ddf_agn_id_set = set(ddf_agn_df['galaxy_id'])
    drop_lens_gals = [idx for idx, gal_id in \
                      enumerate(lens_id_list) if gal_id in ddf_agn_id_set]
    lens_galaxy_df = trim_lens_df.drop(drop_lens_gals).reset_index(drop=True)

    # Load av_mw, rv_mw for truth catalogs
    ebvObj = EBVbase()
    ebv_vals = ebvObj.calculateEbv(interp=True,
                    equatorialCoordinates=np.array([np.radians(lens_galaxy_df['ra']),
                                                    np.radians(lens_galaxy_df['dec'])]))
    rv_mw = 3.1
    av_vals = rv_mw * ebv_vals
    lens_galaxy_df['rv_mw'] = rv_mw
    lens_galaxy_df['av_mw'] = av_vals

    return lens_galaxy_df

def get_healpix_id(ra, dec):
    pix_id = healpy.ang2pix(32, ra, dec, nest=False, lonlat=True)
    return pix_id

def load_dc2_hosts(catalog_version, agn_db, sed_dir):

    # Load DC2 lens and host catalogs
    dc2_reader = DC2Reader(catalog_version)
    host_cat_filters = ['ra < 75.', 'ra > 72.5',
                        'dec < -42.5', 'dec > -45.',
                        'redshift_true > 0.25']
    host_galaxy_df = dc2_reader.load_galaxy_catalog(host_cat_filters)

    # Load DC2 AGN
    host_agn_query = 'ra > 72.5 and ra < 75. and dec > -45 and dec < -42.5 and redshift > 0.25'
    host_agn_df = dc2_reader.load_agn_catalog(agn_db, host_agn_query)

    # Load healpix ids so we can get SED information
    healpix_ids_hosts = get_healpix_id(host_galaxy_df['ra'],
                                       host_galaxy_df['dec'])
    host_galaxy_df['healpix_id'] = healpix_ids_hosts

    # To make SED loading quicker (we have to load an entire healpixel worth of SED
    # info at a time) we will only take the four healpixels with the most galaxies.
    # For more info see notebook matchingLensGalaxies.ipynb

    healpix_query = 'healpix_id < 10203 or healpix_id == 10329 or healpix_id == 10452'
    host_galaxy_df = host_galaxy_df.query(healpix_query).reset_index(drop=True)

    # Join host and agn info
    host_full_df = host_galaxy_df.join(host_agn_df.set_index('galaxy_id'),
                                       on='galaxy_id', rsuffix='_agn')
    host_full_df = host_full_df.rename(columns={'magNorm':'magNorm_agn',
                                                'varParamStr':'varParamStr_agn',
                                                'redshift':'redshift_agn'})
    # Pick out the host galaxies that have an AGN
    # These will be our potential matches for lensed AGN
    agn_host_full_df = host_full_df.query('ra_agn > -99.').reset_index(drop=True)
    # Host galaxies without an AGN are our pool of potential SNe host galaxies
    sne_host_full_df = host_full_df.iloc[np.where(np.isnan(host_full_df['M_i'].values))]

    # Load i-band mags for AGN so we can match to OM10
    lsst_bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
    agn_sed = Sed()
    agn_sed.readSED_flambda(os.path.join(getPackageDir('SIMS_SED_LIBRARY'),
                                         'agnSED', 'agn.spec.gz'))
    mag_i_agn = []
    for agn_data in agn_host_full_df[['magNorm_agn', 'redshift_true']].values:
        agn_mag, agn_redshift = agn_data
        if np.isnan(agn_mag):
            mag_i_agn.append(np.nan)
            continue
        agn_copy = deepcopy(agn_sed)
        flux_norm = getImsimFluxNorm(agn_copy, agn_mag)
        agn_copy.multiplyFluxNorm(flux_norm)
        agn_copy.redshiftSED(agn_redshift, dimming=True)
        mag_i_agn.append(agn_copy.calcMag(lsst_bp_dict['i']))
    agn_host_full_df['mag_i_agn'] = mag_i_agn

    sed_df = pd.DataFrame([], columns=['bulge_av', 'bulge_rv', 'bulge_sed',
                                       'disk_av', 'disk_rv', 'disk_sed',
                                       'galaxy_id',])
    for hpix in np.unique(host_full_df['healpix_id']):

        # This file does not exist. And there are only 19 galaxies in this healpix anyway.
        if hpix == 10572:
            continue

        f = h5py.File(os.path.join(sed_dir, 'sed_fit_%i.h5' % hpix))

        sed_hpix_df = pd.DataFrame([])

        band_list = ['u', 'g', 'r', 'i', 'z', 'y']
        sed_names = f['sed_names'][()]
        for key in list(f.keys()):
            print(key, f[key].len())
            # if (key.endswith('fluxes') or key.endswith('magnorm')):
            if key.endswith('magnorm'):
                key_data = f[key][()]
                for i in range(6):
                    sed_hpix_df[str(key + '_' + band_list[i])] = key_data[i]
            elif key in ['sed_names', 'ra', 'dec']:
                continue
            elif key.endswith('fluxes'):
                continue
            elif key.endswith('sed'):
                sed_hpix_df[key] = [sed_names[idx] for idx in f[key][()]]
            else:
                sed_hpix_df[key] = f[key][()]

        sed_df = pd.concat([sed_df, sed_hpix_df], sort=False)

    sed_df = sed_df.reset_index(drop=True)
    sed_df = sed_df.drop(columns=['redshift'])
    agn_host_join = agn_host_full_df.join(sed_df.set_index('galaxy_id'),
                                          on='galaxy_id')
    sne_host_join = sne_host_full_df.join(sed_df.set_index('galaxy_id'),
                                          on='galaxy_id')

    # Clean up galaxies with bad magnorms
    good_agn_rows = np.where((~np.isnan(agn_host_join['bulge_magnorm_u']) &
                              ~np.isinf(agn_host_join['bulge_magnorm_u']) &
                              ~np.isnan(agn_host_join['disk_magnorm_u']) &
                              ~np.isinf(agn_host_join['disk_magnorm_u'])))[0]
    agn_host_final = agn_host_join.iloc[good_agn_rows].reset_index(drop=True)

    good_sne_rows = np.where((~np.isnan(sne_host_join['bulge_magnorm_u']) &
                              ~np.isinf(sne_host_join['bulge_magnorm_u']) &
                              ~np.isnan(sne_host_join['disk_magnorm_u']) &
                              ~np.isinf(sne_host_join['disk_magnorm_u'])))[0]
    sne_host_final = sne_host_join.iloc[good_sne_rows].reset_index(drop=True)

    return agn_host_final, sne_host_final

def merge_catalog(df_sys, df_img):

    img_pivot = df_img.pivot(index='sysno', columns='imno')
    img_df = pd.DataFrame([])
    img_df['sysno'] = img_pivot.index.values
    img_df['n_img'] = np.sum(~np.isnan(img_pivot['td']), axis=1).values
    img_df['x_img'] = list(img_pivot['x'].values)
    img_df['y_img'] = list(img_pivot['y'].values)
    img_df['t_delay_img'] = list(img_pivot['td'].values)
    img_df['magnification_img'] = list(img_pivot['mu'].values)

    df_merged = df_sys.merge(img_df, on='sysno')

    return df_merged

def fits_to_pandas(input_fits_table):

    cat_dict = {}

    for key in input_fits_table.columns.names:
        cat_vals = input_fits_table[key]
        cat_vals = cat_vals.byteswap().newbyteorder()
        if len(np.shape(cat_vals)) == 1:
            cat_dict[key] = cat_vals
        else:
            cat_dict[key] = list(cat_vals)

    return pd.DataFrame(cat_dict)

def run_dc2_sprinkler(dc2_lenses, dc2_agn_hosts, dc2_sne_hosts,
                      om10_data, glsne_merged_df, output_dir):

    # Match Lenses
    dc2_sprinkler = DC2Sprinkler()

    # Add velocity dispersion estimates to lens cats
    gal_radius = dc2_lenses['morphology/spheroidHalfLightRadius'].values
    gal_radius_arcsec = dc2_lenses['morphology/spheroidHalfLightRadiusArcsec'].values
    sigma_fp = dc2_sprinkler.calc_velocity_dispersion(gal_radius,
                                                      dc2_sprinkler.calc_mu_e(dc2_lenses['mag_true_r_lsst'].values,
                                                                              gal_radius_arcsec,
                                                                              dc2_lenses['redshift_true'].values))
    dc2_lenses['fp_vel_disp'] = sigma_fp



    # Match OM10 Lenses
    om10_match_idx, lens_gal_match_idx, om10_lensid = dc2_sprinkler.match_to_lenscat_agn(sigma_fp,
                                                                                        dc2_lenses['redshift_true'],
                                                                                        om10_data, density=0.09)
    agn_matched_ddf_lenses = dc2_lenses.iloc[lens_gal_match_idx].reset_index(drop=True)
    agn_matched_ddf_lenses['LENSID'] = np.array(om10_data['LENSID'][om10_match_idx], dtype=np.int)
    dc2_lenses_post_agn_matches = dc2_lenses.drop(lens_gal_match_idx).reset_index(drop=True)
    om10_ddf_systems = om10_data[om10_match_idx]

    # Add SNCosmo Parameters
    glsne_merged_df = dc2_sprinkler.add_sncosmo_params(glsne_merged_df)
    # Discard dc2 galaxies that won't match no matter what because they are outside redshift, vel. disp range of lens catalog lenses
    max_z_glsne_lens = np.power(10, np.log10(np.max(glsne_merged_df['zl'])) + 0.03)
    min_vel_disp = np.power(10, np.log10(np.min(glsne_merged_df['sigma'])) - 0.03)
    dc2_lenses_sne_set = dc2_lenses_post_agn_matches.query('redshift_true < %f and fp_vel_disp > %f' % (max_z_glsne_lens,
                                                                                                            min_vel_disp)).reset_index(drop=True)
    glsne_match_idx, lens_gal_sne_match_idx, glsne_sysno = dc2_sprinkler.match_to_lenscat_sne(dc2_lenses_sne_set['fp_vel_disp'].values,
                                                                                              dc2_lenses_sne_set['redshift_true'].values,
                                                                                              glsne_merged_df['zl'].values,
                                                                                              glsne_merged_df['sigma'].values,
                                                                                              glsne_merged_df['sysno'].values,
                                                                                              glsne_merged_df['weight'].values,
                                                                                              density=0.85)
    sne_matched_ddf_lenses = dc2_lenses_sne_set.iloc[lens_gal_sne_match_idx].reset_index(drop=True)
    sne_matched_ddf_lenses['LENSID'] = np.array(glsne_merged_df['sysno'][glsne_match_idx], dtype=np.int)
    glsne_ddf_systems = glsne_merged_df.iloc[glsne_match_idx]

    # Match hosts
    om10_index, agn_host_gal_index, om10_matched_lensid = dc2_sprinkler.match_hosts_om10(dc2_agn_hosts['redshift_true'],
                                                                                         dc2_agn_hosts['mag_i_agn'],
                                                                                         om10_ddf_systems)
    agn_final_ddf_lenses = agn_matched_ddf_lenses.iloc[om10_index]
    agn_final_ddf_hosts = dc2_agn_hosts.iloc[agn_host_gal_index].reset_index(drop=True)
    agn_final_ddf_hosts['LENSID'] = np.array(om10_matched_lensid, dtype=np.int)
    om10_fits_final = om10_ddf_systems[om10_index]
    om10_ddf_final = fits_to_pandas(om10_fits_final)

    glsne_index, sne_host_gal_index, glsne_matched_lensid = dc2_sprinkler.match_hosts_glsne(dc2_sne_hosts['redshift_true'],
                                                                                            dc2_sne_hosts['size_true'],
                                                                                            glsne_ddf_systems['zs'].values,
                                                                                            glsne_ddf_systems['host_reff'],
                                                                                            glsne_ddf_systems['sysno'].values)
    sne_final_ddf_lenses = sne_matched_ddf_lenses.iloc[glsne_index]
    sne_final_ddf_hosts = dc2_sne_hosts.iloc[sne_host_gal_index].reset_index(drop=True)
    sne_final_ddf_hosts['LENSID'] = np.array(glsne_matched_lensid, dtype=np.int)
    glsne_ddf_final = glsne_ddf_systems.iloc[glsne_index]

    agn_final_ddf_lenses = agn_final_ddf_lenses.rename(columns={'shear_1': 'gamma_1',
                                                                'shear_2_phosim':'gamma_2',
                                                                'size_true':'size',
                                                                'size_minor_true':'size_minor',
                                                                'convergence':'kappa',
                                                                'position_angle_true':'position_angle',})
    sne_final_ddf_lenses = sne_final_ddf_lenses.rename(columns={'shear_1': 'gamma_1',
                                                                'shear_2_phosim':'gamma_2',
                                                                'size_true':'size',
                                                                'size_minor_true':'size_minor',
                                                                'convergence':'kappa',
                                                                'position_angle_true':'position_angle',})

    om10_ddf_sprinkler_cat = om10_ddf_final.rename(columns={'ZLENS':'z_lens',
                                                            'PHIE':'phie_lens',
                                                            'GAMMA':'gamma',
                                                            'PHIG':'phi_gamma',
                                                            'ELLIP':'ellip_lens',
                                                            'LENSID':'system_id',
                                                            'NIMG':'n_img',
                                                            'XSRC':'x_src',
                                                            'YSRC':'y_src',
                                                            'XIMG':'x_img',
                                                            'YIMG':'y_img',
                                                            'ZSRC':'z_src',
                                                            'MAG':'magnification_img',
                                                            'DELAY':'t_delay_img'})
    glsne_ddf_final = glsne_ddf_final.rename(columns={'zl':'z_lens',
                                                      'sysno':'system_id',
                                                      'theta_gamma': 'phi_gamma',
                                                      'host_x': 'x_src',
                                                      'host_y': 'y_src',
                                                      'zs':'z_src',
                                                      'e':'ellip_lens',
                                                      'theta_e':'phie_lens'})

    agn_lens_truth, sne_lens_truth = dc2_sprinkler.output_lens_galaxy_truth(agn_final_ddf_lenses, om10_ddf_sprinkler_cat,
                                                                            sne_final_ddf_lenses, glsne_ddf_final,
                                                                            os.path.join(output_dir, 'lens_truth.db'),
                                                                            return_df=True, overwrite_existing=True)

    agn_final_ddf_hosts = agn_final_ddf_hosts.rename(columns={'size_disk_true':'semi_major_axis_disk',
                                                            'size_bulge_true':'semi_major_axis_bulge',
                                                            'size_minor_disk_true':'semi_minor_axis_disk',
                                                            'size_minor_bulge_true':'semi_minor_axis_bulge',
                                                            'size_true': 'semi_major_axis',
                                                            'size_minor_true': 'semi_minor_axis',
                                                            'position_angle_true':'position_angle',
                                                            'disk_av':'av_internal_disk',
                                                            'disk_rv':'rv_internal_disk',
                                                            'bulge_av':'av_internal_bulge',
                                                            'bulge_rv':'rv_internal_bulge',
                                                            'disk_sed':'sed_disk',
                                                            'bulge_sed':'sed_bulge'})

    sne_final_ddf_hosts = sne_final_ddf_hosts.rename(columns={'size_disk_true':'semi_major_axis_disk',
                                                            'size_bulge_true':'semi_major_axis_bulge',
                                                            'size_minor_disk_true':'semi_minor_axis_disk',
                                                            'size_minor_bulge_true':'semi_minor_axis_bulge',
                                                            'size_true': 'semi_major_axis',
                                                            'size_minor_true': 'semi_minor_axis',
                                                            'position_angle_true':'position_angle',
                                                            'disk_av':'av_internal_disk',
                                                            'disk_rv':'rv_internal_disk',
                                                            'bulge_av':'av_internal_bulge',
                                                            'bulge_rv':'rv_internal_bulge',
                                                            'disk_sed':'sed_disk',
                                                            'bulge_sed':'sed_bulge'})

    agn_host_truth, sne_host_truth = dc2_sprinkler.output_host_galaxy_truth(agn_final_ddf_lenses, agn_final_ddf_hosts,
                                                                            om10_ddf_sprinkler_cat,
                                                                            sne_final_ddf_lenses, sne_final_ddf_hosts,
                                                                            glsne_ddf_final,
                                                                            os.path.join(output_dir, 'host_truth.db'),
                                                                            return_df=True, overwrite_existing=True)

    lensed_agn_truth = dc2_sprinkler.output_lensed_agn_truth(agn_final_ddf_hosts,
                                                             agn_final_ddf_lenses,
                                                             om10_ddf_sprinkler_cat,
                                                             os.path.join(output_dir, 'lensed_agn_truth.db'),
                                                             return_df=True, overwrite_existing=True)

    lensed_sne_truth = dc2_sprinkler.output_lensed_sne_truth(sne_final_ddf_hosts,
                                                             sne_final_ddf_lenses,
                                                             glsne_ddf_final,
                                                             os.path.join(output_dir, 'lensed_sne_truth.db'),
                                                             return_df=True, overwrite_existing=True)


if __name__ == '__main__':

    catalog_version = 'cosmoDC2_v1.1.4_image_addon_knots'
    agn_db = '/global/cscratch1/sd/jchiang8/desc/sims_GCRCatSimInterface/work/2020-02-14/agn_cosmoDC2_v1.1.4.db'
    sed_dir = '/global/projecta/projectdirs/lsst/groups/SSim/DC2/cosmoDC2_v1.1.4/sedLookup'

    parser = argparse.ArgumentParser(
        description='Run DC2 Sprinkler and generate truth catalogs.'
    )
    parser.add_argument('--input_dir', type=str, help='Input Data Directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint File Directory')
    parser.add_argument('--output_dir', type=str, help='Output Data Directory')
    args = parser.parse_args()

    # Load DC2 catalogs
    if os.path.exists(os.path.join(args.checkpoint_dir, 'dc2_lens_ckpt.csv')):
        dc2_lenses = pd.read_csv(os.path.join(args.checkpoint_dir, 'dc2_lens_ckpt.csv'))
    else:
        dc2_lenses = load_dc2_lenses(catalog_version, agn_db)
        dc2_lenses.to_csv(os.path.join(args.checkpoint_dir, 'dc2_lens_ckpt.csv'), index=False)

    if (os.path.exists(os.path.join(args.checkpoint_dir, 'dc2_agn_host_ckpt.csv')) &
        os.path.exists(os.path.join(args.checkpoint_dir, 'dc2_sne_host_ckpt.csv'))):
        dc2_agn_hosts = pd.read_csv(os.path.join(args.checkpoint_dir, 'dc2_agn_host_ckpt.csv'))
        dc2_sne_hosts = pd.read_csv(os.path.join(args.checkpoint_dir, 'dc2_sne_host_ckpt.csv'))
    else:
        dc2_agn_hosts, dc2_sne_hosts = load_dc2_hosts(catalog_version, agn_db, sed_dir)
        dc2_agn_hosts.to_csv(os.path.join(args.checkpoint_dir, 'dc2_agn_host_ckpt.csv'), index=False)
        dc2_sne_hosts.to_csv(os.path.join(args.checkpoint_dir, 'dc2_sne_host_ckpt.csv'), index=False)

    # Load AGN Lens Cat
    om10_hdu = fits.open(os.path.join(args.input_dir, 'om10_qso_mock.fits'))
    om10_data = om10_hdu[1].data
    # Only keep om10 systems where the host galaxy is within the redshift range of cosmoDC2
    om10_data = om10_data[np.where(om10_data['ZSRC'] <= 3.0)]

    # Load SNe Lens Cat
    glsne_df_system = pd.read_hdf(os.path.join(args.input_dir, 'glsne_dc2_v2.h5'), key='system')
    glsne_df_image = pd.read_hdf(os.path.join(args.input_dir, 'glsne_dc2_v2.h5'), key='image')
    glsne_merged_df = merge_catalog(glsne_df_system, glsne_df_image)
    # Remove objects where the host galaxy is more than 1.25 arcseconds
    # from the source so we can image it with postage stamp code
    glsne_query = 'host_x < 1.25 and host_y < 1.25 and host_x > -1.25 and host_y > -1.25'
    glsne_merged_df = glsne_merged_df.query(glsne_query).reset_index(drop=True)

    # Run Match and Truth Catalog Generation
    run_dc2_sprinkler(dc2_lenses, dc2_agn_hosts, dc2_sne_hosts,
                      om10_data, glsne_merged_df, args.output_dir)
