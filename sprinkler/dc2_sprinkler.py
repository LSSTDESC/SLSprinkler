import os
import json
import pandas as pd
import numpy as np
import sncosmo
from astropy.cosmology import FlatLambdaCDM
from sqlalchemy import create_engine
from .base_sprinkler import BaseSprinkler
from lsst.sims.photUtils import Sed, BandpassDict, getImsimFluxNorm, Bandpass
from lsst.utils import getPackageDir
from copy import deepcopy

__all__ = ['DC2Sprinkler']


class DC2Sprinkler(BaseSprinkler):

    def match_to_lenscat_agn(self, vel_disp, redshift, om10_array, density=1.0):

        """
        Match DC2 galaxies to lens galaxies in OM10 based upon velocity dispersion and redshift.
        We set the threshold for a match to be 0.03 in dex in each parameter.

        Parameters
        ----------

        vel_disp: numpy.ndarray
        Velocity dispersion (km/s) values for each cosmoDC2 galaxy that is a potential lens galaxy

        redshift: numpy.ndarray
        Redshifts of potential lens galaxies from cosmoDC2

        om10_array: FITS recarray
        OM10 Lens Catalog

        Returns
        -------

        om10_idx: List of ints
            Indices of matches in OM10 catalog from input `om10_array`

        lens_gal_idx: List of ints
            Indices of corresponding cosmoDC2 galaxies from catalog that provided input `vel_disp` and `redshift`

        om10_lens_ids: List of ints
            OM10 LENSID values for systems matched is corresponding order as other outputs
        """

        lens_gal_idx = []
        om10_idx = []

        i = 0
        successful_matches = 0

        for row in om10_array:

            rand_state = np.random.RandomState(row['LENSID'])

            if i % 500 == 0:
                print("Matched %i out of %i possible OM10 systems so far. Total Catalog Length: %i" %
                    (successful_matches, i, len(om10_array)))
            i += 1

            lens_z = row['ZLENS']
            log_lens_z = np.log10(lens_z)

            lens_sigma = row['VELDISP']
            log_lens_sigma = np.log10(lens_sigma)

            match_prob = rand_state.uniform()

            # adjust matching probability for low z lenses
            # to get closer match to overall om10 in redshift and vel. disp.
            if lens_z < 0.3:
                match_prob *= 0.2
            if lens_sigma >= 225.:
                match_prob *= 0.5

            if match_prob > density:
                continue

            match_idx = (np.where((np.abs(log_lens_sigma - np.log10(vel_disp)) < 0.03) &
                                (np.abs(log_lens_z - np.log10(redshift)) < 0.03)))[0]
            # Avoid duplicates
            match_idx_keep = [w for w in match_idx if w not in lens_gal_idx]
            if len(match_idx_keep) == 0:
                continue
            # Randomly choose one of the matches
            matched_lens_idx = rand_state.choice(match_idx_keep)

            lens_gal_idx.append(matched_lens_idx)
            om10_idx.append(i-1)
            successful_matches += 1

        om10_lens_ids = om10_array['LENSID'][om10_idx]

        return om10_idx, lens_gal_idx, om10_lens_ids

    def match_to_lenscat_sne(self, vel_disp, redshift, glsne_cat_zl,
                             glsne_cat_sigma, glsne_cat_sysno,
                             glsne_weights, density=1.0):

        """
        Match DC2 galaxies to lens galaxies in OM10 based upon velocity dispersion and redshift.
        We set the threshold for a match to be 0.03 in dex in each parameter.

        Parameters
        ----------

        vel_disp: numpy.ndarray
        Velocity dispersion (km/s) values for each cosmoDC2 galaxy that is a potential lens galaxy

        redshift: numpy.ndarray
        Redshifts of potential lens galaxies from cosmoDC2

        glsne_cat: pandas dataframe
        Catalog of Strongly Lensed SNe systems from Goldstein et al. 2019

        Returns
        -------

        glsne_idx: List of ints
            Indices of matches in glsne catalog from input `glsne_cat`

        lens_gal_idx: List of ints
            Indices of corresponding cosmoDC2 galaxies from catalog that provided input `vel_disp` and `redshift`

        glsne_lens_ids: List of ints
            glsne catalog `sysno` values for systems matched is corresponding order as other outputs
        """

        lens_gal_idx = []
        glsne_idx = []

        normed_weights = glsne_weights/np.max(glsne_weights)

        i = 0
        successful_matches = 0

        for lens_z, lens_sigma, lens_sysno, sys_weight in zip(glsne_cat_zl, glsne_cat_sigma, glsne_cat_sysno, normed_weights):

            rand_state = np.random.RandomState(lens_sysno)

            if i % 5000 == 0:
                print("Matched %i out of %i possible GLSNe systems so far. Total Catalog Length: %i" %
                    (successful_matches, i, len(glsne_cat_zl)))
            i += 1

            log_lens_z = np.log10(lens_z)

            log_lens_sigma = np.log10(lens_sigma)

            match_prob = rand_state.uniform()
            match_density = density * sys_weight * 10
    #         if lens_sigma > 190.:
    #             match_prob = match_density - 0.01
    #         if lens_z < 0.4:
    #             match_prob = density - 0.01
    #         if lens_sigma < 130.:
    #             match_prob *= 2.

            if (match_prob > match_density):
                continue

            match_idx = (np.where((np.abs(log_lens_sigma - np.log10(vel_disp)) < 0.03) &
                                (np.abs(log_lens_z - np.log10(redshift)) < 0.03)))[0]
            # Avoid duplicates
            match_idx_keep = [w for w in match_idx if w not in lens_gal_idx]
            if len(match_idx_keep) == 0:
                continue
            # Randomly choose one of the matches
            matched_lens_idx = rand_state.choice(match_idx_keep)

            lens_gal_idx.append(matched_lens_idx)
            glsne_idx.append(i-1)
            successful_matches += 1

        glsne_lens_ids = glsne_cat_sysno[glsne_idx]

        return glsne_idx, lens_gal_idx, glsne_lens_ids

    def add_sncosmo_params(self, glsne_merged_df):

        source = sncosmo.get_source('salt2-extended')
        model = sncosmo.Model(source=source)

        # Use cosmoDC2 settings
        cosmo = FlatLambdaCDM(H0=71, Om0=0.265, Tcmb0=0, Neff=3.04, m_nu=None, Ob0=0.045)

        x0 = []
        for sn_row_zs, sn_row_mb in zip(glsne_merged_df['zs'].values, glsne_merged_df['MB'].values):
            z = sn_row_zs
            MB = sn_row_mb
            model.set(z=z)
            model.set_source_peakabsmag(MB, 'bessellb', 'ab', cosmo=cosmo)
            x0.append(model.get('x0'))

        glsne_merged_df['x0'] = x0
        glsne_merged_df['x1'] = 1.
        glsne_merged_df['c'] = 0.

        return glsne_merged_df

    def match_hosts_om10(self, redshift, agn_i_mag, om10_systems):

        """
        Match host galaxies to OM10 systems based upon redshift and i-band AGN magnitude.

        Parameters
        ----------

        redshift: numpy ndarray
            Array of DC2 potential host redshifts

        agn_i_mag: numpy ndarray
            Array of DC2 potential AGN i-band magnitudes in same order as redshift

        om10_systems: om10 fits data table
            The subselection of OM10 that matched to a lens galaxy already

        Returns
        -------

        om10_idx: list
            Row index in input `om10_systems` that matched to a DC2 host galaxy

        host_gal_idx: list
            Row index in the input DC2 `redshift` and `agn_i_mag` arrays that matched to an om10 system in
            the same order as `om10_idx`

        om10_system_ids: list
            `LENSID` values of the om10 systems that matched to a DC2 host galaxy
        """

        i = 0
        om10_idx = []
        host_gal_idx = []

        for om10_row in om10_systems:
            rand_state = np.random.RandomState(om10_row['LENSID'])
            log_z_om10 = np.log10(om10_row['ZSRC'])
            imag_om10 = om10_row['MAGI_IN']

            matches = np.where((np.abs(np.log10(redshift) - log_z_om10) < 0.05) &
                            (np.abs(agn_i_mag - imag_om10) < 0.05))[0]
            keep_matches = [w for w in matches if w not in host_gal_idx]
            if len(keep_matches) > 0:
                gal_match = rand_state.choice(keep_matches)
                om10_idx.append(i)
                host_gal_idx.append(gal_match)

            i += 1

        om10_system_ids = om10_systems['LENSID'][om10_idx]

        return om10_idx, host_gal_idx, om10_system_ids

    def match_hosts_glsne(self, redshift, host_size, glsne_redshifts,
                          glsne_host_size, glsne_sysno):

        """
        Match host galaxies to GLSNE systems based upon redshift

        Parameters
        ----------

        redshift: numpy ndarray
            Array of DC2 potential host redshifts

        glsne_redshifts: numpy ndarray
            The redshifts from the subselection of GLSNe that matched to a lens galaxy already

        glsne_sysno: numpy ndarray
            The system numbers of the subselection of GLSNe that matched to a lens galaxy already

        Returns
        -------

        glsne_idx: list
            Row index in input `glsne_redshift` and `glsne_sysno` that matched to a DC2 host galaxy

        host_gal_idx: list
            Row index in the input DC2 `redshift` array that matched to a glsne system in
            the same order as `glsne_idx`

        glsne_system_ids: list
            `glsne_sysno` values of the glsne systems that matched to a DC2 host galaxy
        """

        i = 0
        glsne_idx = []
        host_gal_idx = []

        for glsne_z, glsne_size, row_sysno in zip(glsne_redshifts, glsne_host_size, glsne_sysno):
            rand_state = np.random.RandomState(row_sysno)
            log_z_glsne = np.log10(glsne_z)
            log_size_glsne = np.log10(glsne_size)

            if i % 50 == 0:
                print(i)

            matches = np.where((np.abs(np.log10(redshift) - log_z_glsne) < 0.05) &
                            (np.abs(np.log10(host_size) - log_size_glsne) < 0.05))[0]
            keep_matches = [w for w in matches if w not in host_gal_idx]
            if len(keep_matches) > 0:
                gal_match = rand_state.choice(keep_matches)
                glsne_idx.append(i)
                host_gal_idx.append(gal_match)

            i += 1

            glsne_system_ids = glsne_sysno[glsne_idx]

        return glsne_idx, host_gal_idx, glsne_system_ids

    def add_flux(self, sed_name, redshift, magnorm_dict, av, rv, bp_dict=None):

        if bp_dict is None:
            bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

        result_dict_no_mw = {}
        result_dict_mw = {}

        sed_obj = Sed()
        sed_obj.readSED_flambda(os.path.join(os.environ['SIMS_SED_LIBRARY_DIR'], sed_name))
        a_x, b_x = sed_obj.setupCCM_ab()
        for bandpass_name in bp_dict.keys():
            sed_copy = deepcopy(sed_obj)
            flux_norm = getImsimFluxNorm(sed_copy, magnorm_dict[bandpass_name])
            sed_copy.multiplyFluxNorm(flux_norm)
            sed_copy.redshiftSED(redshift, dimming=True)
            band_flux = sed_copy.calcFlux(bp_dict[bandpass_name])
            result_dict_no_mw[bandpass_name] = band_flux
            sed_copy.addDust(a_x, b_x, A_v=av, R_v=rv)
            band_flux_mw = sed_copy.calcFlux(bp_dict[bandpass_name])
            result_dict_mw[bandpass_name] = band_flux_mw

        return result_dict_no_mw, result_dict_mw

    def create_lens_truth_dataframe(self, matched_lenses, matched_sys_cat, id_type_prefix):

        """
        Create the final properly formatted lens galaxy truth catalog.

        Parameters
        ----------
        matched_lenses: pandas dataframe
            Dataframe of matched DC2 lenses

        matched_sys_cat: pandas dataframe
            Dataframe of matched lens catalog systems

        id_type_prefix: str
            Ids to match systems will be (id_type_prefix)_(lens sys id number)
            Example: `GLAGN_0`

        Returns
        -------
        lens_df: pandas dataframe
            Return the pandas dataframe format of the truth catalog.
        """

        new_entries = []

        for i in range(len(matched_sys_cat)):

            gal_id = matched_lenses.iloc[i]['galaxy_id']
            new_sys_id_num = i
            new_sys_id = '%s_%i' % (id_type_prefix, new_sys_id_num)
            # The newly inserted lens galaxy keeps the old gal_id
            unique_id = gal_id

            ra_lens = matched_lenses.iloc[i]['ra']
            dec_lens = matched_lenses.iloc[i]['dec']

            redshift = matched_sys_cat.iloc[i]['z_lens']
            shear_1_dc2 = matched_lenses.iloc[i]['gamma_1']
            shear_2_dc2 = matched_lenses.iloc[i]['gamma_2']
            kappa_dc2 = matched_lenses.iloc[i]['kappa']
            gamma_lenscat = matched_sys_cat.iloc[i]['gamma']
            phi_gamma_lenscat = matched_sys_cat.iloc[i]['phi_gamma']
            shear_1_lenscat = gamma_lenscat * np.cos(2 * phi_gamma_lenscat)
            shear_2_lenscat = gamma_lenscat * np.sin(2 * phi_gamma_lenscat)
            sindex_lens = 4

    #         major_axis_lens = matched_sys_cat.iloc[i]['reff_lens'] / \
    #                             np.sqrt(1 - matched_sys_cat.iloc[i]['ellip_lens'])
    #         minor_axis_lens = matched_sys_cat.iloc[i]['reff_lens'] * \
    #                             np.sqrt(1 - matched_sys_cat.iloc[i]['ellip_lens'])
    #         position_angle = matched_sys_cat.iloc[i]['phie_lens']*(-1.0)*np.pi/180.0
            major_axis_lens = matched_lenses.iloc[i]['size']
            minor_axis_lens = matched_lenses.iloc[i]['size_minor']
            position_angle = matched_lenses.iloc[i]['position_angle']

            # Change dc2 ellip [(1-q)/(1+q)] to om10 ellipticity which is (1-q)
            q = minor_axis_lens/major_axis_lens
            ellip_dc2 = 1.0 - q
            ellip_lens = matched_sys_cat.iloc[i]['ellip_lens']
            # Convert DC2 position angle to `phie` which starts from y-axis
            phie_dc2 = 0.5*position_angle - 90
            phie_lens = matched_sys_cat.iloc[i]['phie_lens']

            av_mw = matched_lenses.iloc[i]['av_mw']
            rv_mw = matched_lenses.iloc[i]['rv_mw']

            vel_disp_lens = matched_lenses.iloc[i]['fp_vel_disp']

            cat_sys_id = matched_sys_cat.iloc[i]['system_id']

            new_row = [unique_id, ra_lens, dec_lens,
                    redshift, shear_1_dc2, shear_2_dc2, kappa_dc2,
                    gamma_lenscat, phi_gamma_lenscat,
                    shear_1_lenscat, shear_2_lenscat, sindex_lens,
                    major_axis_lens, minor_axis_lens,
                    position_angle, ellip_dc2, ellip_lens, phie_dc2, phie_lens,
                    av_mw, rv_mw, vel_disp_lens,
                    cat_sys_id, new_sys_id]

            new_entries.append(new_row)

        lens_df = pd.DataFrame(new_entries,
                            columns=['unique_id', 'ra_lens', 'dec_lens',
                                        'redshift', 'shear_1_cosmodc2', 'shear_2_cosmodc2',
                                        'kappa_cosmodc2', 'gamma_lenscat', 'phig_lenscat',
                                        'shear_1_lenscat', 'shear_2_lenscat',
                                        'sindex_lens', 'major_axis_lens',
                                        'minor_axis_lens', 'position_angle', 'ellip_cosmodc2',
                                        'ellip_lens', 'phie_cosmodc2', 'phie_lens', 'av_mw', 'rv_mw',
                                        'vel_disp_lenscat',
                                        'lens_cat_sys_id', 'dc2_sys_id'])

        return lens_df

    def output_lens_galaxy_truth(self, matched_agn_lens, matched_agn_sys,
                                 matched_sne_lens, matched_sne_sys, out_file,
                                 return_df=False, overwrite_existing=False):

        """
        Output sqlite truth catalogs for foreground lens galaxies for
        lensed AGN and SNe.

        Parameters
        ----------
        matched_agn_lens: pandas dataframe
            Dataframe of matched DC2 lenses for lensed AGN systems

        matched_agn_sys: pandas dataframe
            Dataframe of matched lens catalog systems for lensed AGN systems

        matched_sne_lens: pandas dataframe
            Dataframe of matched DC2 lenses for lensed SNe systems

        matched_sne_sys: pandas dataframe
            Dataframe of matched lens catalog systems for lensed SNe systems

        out_file: str
            Filename of sqlite truth catalog for lens galaxies

        return_df: bool, deafult=False
            Return the dataframes of the lens galaxy truth catalogs.

        overwrite_existing: bool, default=False
            Overwrite existing catalog

        Returns
        -------
        agn_lens_df: pandas dataframe
            Pandas dataframe format of the truth catalog for the lens galaxies in lensed AGN systems.

        sne_lens_df: pandas dataframe
            Pandas dataframe format of the truth catalog for the lens galaxies in lensed SNe systems.
        """

        agn_lens_df = self.create_lens_truth_dataframe(matched_agn_lens,
                                                       matched_agn_sys, 'GLAGN')
        sne_lens_df = self.create_lens_truth_dataframe(matched_sne_lens,
                                                       matched_sne_sys, 'GLSNE')

        if overwrite_existing is True and os.path.exists(out_file):
            os.remove(out_file)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_lens_df.to_sql('agn_lens', con=engine)
        sne_lens_df.to_sql('sne_lens', con=engine)

        if return_df is True:
            return agn_lens_df, sne_lens_df

    def merge_bandpass_columns(self, df_merged, label):

        labelled_columns = ['%s_u' %label,
                            '%s_g' %label,
                            '%s_r' %label,
                            '%s_i' %label,
                            '%s_z' %label,
                            '%s_y' %label]

        label_array = df_merged[labelled_columns].values
        label_dict = {x: y for x, y in zip(['u', 'g', 'r', 'i', 'z', 'y'], df_merged[labelled_columns])}

        return label_array, label_dict

    def create_host_truth_dataframe(self, matched_lenses, matched_hosts,
                                    matched_sys_cat, id_type_prefix):

        """
        Create the final properly formatted lens galaxy truth catalog.

        Parameters
        ----------
        matched_lenses: pandas dataframe
            Dataframe of matched DC2 lenses

        matched_hosts: pandas dataframe
            Dataframe of matched DC2 hosts

        matched_sys_cat: pandas dataframe
            Dataframe of matched lens catalog systems

        id_type_prefix: str
            Ids to match systems will be (id_type_prefix)_(lens sys id number)
            Example: `GLAGN_0`

        Returns
        -------
        host_df: pandas dataframe
            Return the pandas dataframe format of the truth catalog.
        """


        new_entries = []

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):

                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id_num = i
                new_sys_id = '%s_%i' % (id_type_prefix, new_sys_id_num)
                image_number = j
                unique_id = '%s_host_%i_%i' % (id_type_prefix, new_sys_id_num, image_number)

                ra_lens = matched_lenses.iloc[i]['ra']
                dec_lens = matched_lenses.iloc[i]['dec']
                id_lens = matched_lenses.iloc[i]['galaxy_id']

                x_src = matched_sys_cat.iloc[i]['x_src']
                y_src = matched_sys_cat.iloc[i]['y_src']
                x_img = matched_sys_cat.iloc[i]['x_img'][j]
                y_img = matched_sys_cat.iloc[i]['y_img'][j]

                delta_ra_unlensed = x_src / 3600.0
                delta_dec_unlensed = y_src / 3600.0
                ra_host_unlensed = ra_lens + delta_ra_unlensed/np.cos(np.radians(dec_lens))
                dec_host_unlensed = dec_lens + delta_dec_unlensed

                delta_ra_lensed = x_img / 3600.0
                delta_dec_lensed = y_img / 3600.0
                ra_host_lensed = ra_lens + delta_ra_lensed/np.cos(np.radians(dec_lens))
                dec_host_lensed = dec_lens + delta_dec_lensed

                redshift = matched_sys_cat.iloc[i]['z_src']
                shear_1 = 0. #matched_hosts.iloc[i]['gamma_1']
                shear_2 = 0. #matched_hosts.iloc[i]['gamma_2']
                kappa = 0. #matched_hosts.iloc[i]['kappa']
                sindex_bulge = 4
                sindex_disk = 1

                mag_adjust = 2.5*np.log10(np.abs(matched_sys_cat.iloc[i]['magnification_img'][j]))
                magnorm_disk_array, magnorm_disk_dict = self.merge_bandpass_columns(matched_hosts.iloc[i], 'disk_magnorm')
                # Postage stamp code expects unlensed mag
                magnorm_disk = magnorm_disk_array #- mag_adjust
                magnorm_disk_u = magnorm_disk[0]
                magnorm_disk_g = magnorm_disk[1]
                magnorm_disk_r = magnorm_disk[2]
                magnorm_disk_i = magnorm_disk[3]
                magnorm_disk_z = magnorm_disk[4]
                magnorm_disk_y = magnorm_disk[5]

                disk_flux_no_mw, disk_flux_mw = self.add_flux(matched_hosts.iloc[i]['sed_disk'][2:-1],
                                                        redshift,
                                                        magnorm_disk_dict, matched_lenses.iloc[i]['av_mw'],
                                                        matched_lenses.iloc[i]['rv_mw'], bp_dict=bp_dict)

                magnorm_bulge_array, magnorm_bulge_dict = self.merge_bandpass_columns(matched_hosts.iloc[i], 'bulge_magnorm')
                magnorm_bulge = magnorm_bulge_array #- mag_adjust
                magnorm_bulge_u = magnorm_bulge[0]
                magnorm_bulge_g = magnorm_bulge[1]
                magnorm_bulge_r = magnorm_bulge[2]
                magnorm_bulge_i = magnorm_bulge[3]
                magnorm_bulge_z = magnorm_bulge[4]
                magnorm_bulge_y = magnorm_bulge[5]

                bulge_flux_no_mw, bulge_flux_mw = self.add_flux(matched_hosts.iloc[i]['sed_bulge'][2:-1],
                                                        redshift,
                                                        magnorm_bulge_dict, matched_lenses.iloc[i]['av_mw'],
                                                        matched_lenses.iloc[i]['rv_mw'], bp_dict=bp_dict)

                flux_u = bulge_flux_mw['u'] + disk_flux_mw['u']
                flux_g = bulge_flux_mw['g'] + disk_flux_mw['g']
                flux_r = bulge_flux_mw['r'] + disk_flux_mw['r']
                flux_i = bulge_flux_mw['i'] + disk_flux_mw['i']
                flux_z = bulge_flux_mw['z'] + disk_flux_mw['z']
                flux_y = bulge_flux_mw['y'] + disk_flux_mw['y']

                flux_u_noMW = bulge_flux_no_mw['u'] + disk_flux_no_mw['u']
                flux_g_noMW = bulge_flux_no_mw['g'] + disk_flux_no_mw['g']
                flux_r_noMW = bulge_flux_no_mw['r'] + disk_flux_no_mw['r']
                flux_i_noMW = bulge_flux_no_mw['i'] + disk_flux_no_mw['i']
                flux_z_noMW = bulge_flux_no_mw['z'] + disk_flux_no_mw['z']
                flux_y_noMW = bulge_flux_no_mw['y'] + disk_flux_no_mw['y']

                major_axis_disk = matched_hosts.iloc[i]['semi_major_axis_disk']
                major_axis_bulge = matched_hosts.iloc[i]['semi_major_axis_bulge']
                minor_axis_disk = matched_hosts.iloc[i]['semi_minor_axis_disk']
                minor_axis_bulge = matched_hosts.iloc[i]['semi_minor_axis_bulge']
                major_axis = matched_hosts.iloc[i]['semi_major_axis']
                minor_axis = matched_hosts.iloc[i]['semi_minor_axis']
                position_angle = matched_hosts.iloc[i]['position_angle']

                av_internal_disk = matched_hosts.iloc[i]['av_internal_disk']
                av_internal_bulge = matched_hosts.iloc[i]['av_internal_bulge']
                rv_internal_disk = matched_hosts.iloc[i]['rv_internal_disk']
                rv_internal_bulge = matched_hosts.iloc[i]['rv_internal_bulge']
                av_mw = matched_lenses.iloc[i]['av_mw']
                rv_mw = matched_lenses.iloc[i]['rv_mw']

                sed_disk_host = matched_hosts.iloc[i]['sed_disk']
                sed_bulge_host = matched_hosts.iloc[i]['sed_bulge']

                cat_sys_id = matched_sys_cat.iloc[i]['system_id']

                new_row = [unique_id, x_src, y_src, x_img, y_img,
                        ra_lens, dec_lens, ra_host_unlensed, dec_host_unlensed,
                        ra_host_lensed, dec_host_lensed,
                        magnorm_disk_u, magnorm_disk_g, magnorm_disk_r,
                        magnorm_disk_i, magnorm_disk_z, magnorm_disk_y,
                        magnorm_bulge_u, magnorm_bulge_g, magnorm_bulge_r,
                        magnorm_bulge_i, magnorm_bulge_z, magnorm_bulge_y,
                        flux_u, flux_g, flux_r, flux_i, flux_z, flux_y,
                        flux_u_noMW, flux_g_noMW, flux_r_noMW, flux_i_noMW, flux_z_noMW, flux_y_noMW,
                        redshift, shear_1, shear_2, kappa, sindex_bulge, sindex_disk,
                        major_axis_disk, major_axis_bulge, minor_axis_disk, minor_axis_bulge,
                        major_axis, minor_axis,
                        position_angle, av_internal_disk, av_internal_bulge, rv_internal_disk,
                        rv_internal_bulge, av_mw, rv_mw, sed_disk_host, sed_bulge_host,
                        gal_id, id_lens, cat_sys_id, new_sys_id, image_number]

                new_entries.append(new_row)

        host_df = pd.DataFrame(new_entries,
                            columns=['unique_id', 'x_src', 'y_src', 'x_img', 'y_img',
                                        'ra_lens', 'dec_lens', 'ra_host_unlensed', 'dec_host_unlensed',
                                        'ra_host_lensed', 'dec_host_lensed',
                                        'magnorm_disk_u', 'magnorm_disk_g', 'magnorm_disk_r',
                                        'magnorm_disk_i', 'magnorm_disk_z', 'magnorm_disk_y',
                                        'magnorm_bulge_u', 'magnorm_bulge_g', 'magnorm_bulge_r',
                                        'magnorm_bulge_i', 'magnorm_bulge_z', 'magnorm_bulge_y',
                                        'flux_u', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'flux_y',
                                        'flux_u_noMW', 'flux_g_noMW', 'flux_r_noMW', 'flux_i_noMW', 'flux_z_noMW', 'flux_y_noMW',
                                        'redshift', 'shear_1', 'shear_2',
                                        'kappa', 'sindex_bulge', 'sindex_disk',
                                        'major_axis_disk', 'major_axis_bulge', 'minor_axis_disk',
                                        'minor_axis_bulge', 'semi_major_axis', 'semi_minor_axis', 'position_angle',
                                        'av_internal_disk', 'av_internal_bulge',
                                        'rv_internal_disk', 'rv_internal_bulge',
                                        'av_mw', 'rv_mw', 'sed_disk_host',
                                        'sed_bulge_host', 'original_gal_id', 'lens_gal_id',
                                        'lens_cat_sys_id', 'dc2_sys_id', 'image_number'])

        return host_df

    def output_host_galaxy_truth(self, matched_agn_lens, matched_agn_hosts, matched_agn_sys,
                                 matched_sne_lens, matched_sne_hosts, matched_sne_sys, out_file,
                                 return_df=False, overwrite_existing=False):

        """
        Output sqlite truth catalogs for foreground lens galaxies for
        lensed AGN and SNe.

        Parameters
        ----------
        matched_agn_lens: pandas dataframe
            Dataframe of matched DC2 lenses for lensed AGN systems

        matched_agn_hosts: pandas dataframe
            Dataframe of matched DC2 hosts for lensed AGN systems

        matched_agn_sys: pandas dataframe
            Dataframe of matched lens catalog systems for lensed AGN systems

        matched_sne_lens: pandas dataframe
            Dataframe of matched DC2 lenses for lensed SNe systems

        matched_sne_hosts: pandas dataframe
            Dataframe of matched DC2 hosts for lensed SNe systems

        matched_sne_sys: pandas dataframe
            Dataframe of matched lens catalog systems for lensed SNe systems

        out_file: str
            Filename of sqlite truth catalog for lens galaxies

        return_df: bool, deafult=False
            Return the dataframes of the lens galaxy truth catalogs.

        overwrite_existing: bool, default=False
            Overwrite existing catalog

        Returns
        -------
        agn_host_df: pandas dataframe
            Pandas dataframe format of the truth catalog for the host galaxies in lensed AGN systems.

        sne_host_df: pandas dataframe
            Pandas dataframe format of the truth catalog for the host galaxies in lensed SNe systems.
        """

        agn_host_df = self.create_host_truth_dataframe(matched_agn_lens, matched_agn_hosts,
                                                       matched_agn_sys, 'GLAGN')
        sne_host_df = self.create_host_truth_dataframe(matched_sne_lens, matched_sne_hosts,
                                                       matched_sne_sys, 'GLSNE')

        if overwrite_existing is True and os.path.exists(out_file):
            os.remove(out_file)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_host_df.to_sql('agn_hosts', con=engine)
        sne_host_df.to_sql('sne_hosts', con=engine)

        if return_df is True:
            return agn_host_df, sne_host_df

    def output_lensed_agn_truth(self, matched_hosts, matched_lenses,
                                matched_sys_cat, out_file,
                                return_df=True,
                                overwrite_existing=False):

        """
        Create the final properly formatted lens galaxy truth catalog.

        Parameters
        ----------
        matched_lenses: pandas dataframe
            Dataframe of matched DC2 lenses

        matched_sys_cat: pandas dataframe
            Dataframe of matched lens catalog systems

        truth_cat_filename: str
            Filename for sqlite truth catalog

        return_df: bool, default=True
            Option to return dataframe after saving to sqlite file

        overwrite_existing: bool, default=False

        Returns
        -------
        lens_df: pandas dataframe
            Return the pandas dataframe format of the truth catalog.
        """

        new_entries = []

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):

                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id_num = i
                new_sys_id = 'GLAGN_%i' % new_sys_id_num
                image_number = j
                gal_unique_id = 'GLAGN_agn_%i_%i' % (new_sys_id_num, image_number)

                ra_lens = matched_lenses.iloc[i]['ra']
                dec_lens = matched_lenses.iloc[i]['dec']
                id_lens = matched_lenses.iloc[i]['galaxy_id']
                x_src = matched_sys_cat.iloc[i]['x_src']
                y_src = matched_sys_cat.iloc[i]['y_src']
                x_img = matched_sys_cat.iloc[i]['x_img'][j]
                y_img = matched_sys_cat.iloc[i]['y_img'][j]
                delta_ra = matched_sys_cat.iloc[i]['x_img'][j] / 3600.0
                delta_dec = matched_sys_cat.iloc[i]['y_img'][j] / 3600.0
                ra = ra_lens + delta_ra/np.cos(np.radians(dec_lens))
                dec = dec_lens + delta_dec

                redshift = matched_sys_cat.iloc[i]['z_src']
                t_delay = matched_sys_cat.iloc[i]['t_delay_img'][j]

                magnorm = matched_hosts.iloc[i]['magNorm_agn']
                mag = matched_sys_cat.iloc[i]['magnification_img'][j]

                magnorm_dict = {x: magnorm for x in ['u', 'g', 'r', 'i', 'z', 'y']}

                agn_flux_no_mw, agn_flux_mw = self.add_flux('agnSED/agn.spec.gz',
                                                    redshift,
                                                    magnorm_dict, matched_lenses.iloc[i]['av_mw'],
                                                    matched_lenses.iloc[i]['rv_mw'], bp_dict=bp_dict)

                agn_var_param = json.loads(matched_hosts.iloc[i]['varParamStr_agn'])['p']
                seed = agn_var_param['seed']
                agn_tau_u = agn_var_param['agn_tau_u']
                agn_tau_g = agn_var_param['agn_tau_g']
                agn_tau_r = agn_var_param['agn_tau_r']
                agn_tau_i = agn_var_param['agn_tau_i']
                agn_tau_z = agn_var_param['agn_tau_z']
                agn_tau_y = agn_var_param['agn_tau_y']
                agn_sf_u = agn_var_param['agn_sf_u']
                agn_sf_g = agn_var_param['agn_sf_g']
                agn_sf_r = agn_var_param['agn_sf_r']
                agn_sf_i = agn_var_param['agn_sf_i']
                agn_sf_z = agn_var_param['agn_sf_z']
                agn_sf_y = agn_var_param['agn_sf_y']

                av_mw = matched_lenses.iloc[i]['av_mw']
                rv_mw = matched_lenses.iloc[i]['rv_mw']

                cat_sys_id = matched_sys_cat.iloc[i]['system_id']

                new_row = [gal_unique_id, ra, dec, x_src, y_src, x_img, y_img,
                        redshift, t_delay, magnorm,
                        agn_flux_mw['u'], agn_flux_mw['g'], agn_flux_mw['r'],
                        agn_flux_mw['i'], agn_flux_mw['z'], agn_flux_mw['y'],
                        agn_flux_no_mw['u'], agn_flux_no_mw['g'], agn_flux_no_mw['r'],
                        agn_flux_no_mw['i'], agn_flux_no_mw['z'], agn_flux_no_mw['y'],
                        mag, seed, agn_tau_u, agn_tau_u, agn_tau_u,
                        agn_tau_u, agn_tau_u, agn_tau_u,
                        agn_sf_u, agn_sf_g, agn_sf_r,
                        agn_sf_i, agn_sf_z, agn_sf_y,
                        av_mw, rv_mw, id_lens, new_sys_id,
                        cat_sys_id, image_number]

                new_entries.append(new_row)

        agn_df = pd.DataFrame(new_entries,
                            columns=['unique_id', 'ra', 'dec', 'x_agn', 'y_agn',
                                    'x_img', 'y_img',
                                    'redshift', 't_delay', 'magnorm',
                                    'flux_u_agn', 'flux_g_agn', 'flux_r_agn',
                                    'flux_i_agn', 'flux_z_agn', 'flux_y_agn',
                                    'flux_u_agn_noMW', 'flux_g_agn_noMW',
                                    'flux_r_agn_noMW', 'flux_i_agn_noMW',
                                    'flux_z_agn_noMW', 'flux_y_agn_noMW',
                                    'magnification', 'seed', 'agn_tau_u',
                                    'agn_tau_g', 'agn_tau_r',
                                    'agn_tau_i', 'agn_tau_z', 'agn_tau_y',
                                    'agn_sf_u', 'agn_sf_g', 'agn_sf_r',
                                    'agn_sf_i', 'agn_sf_z', 'agn_sf_y',
                                    'av_mw', 'rv_mw', 'lens_id', 'dc2_sys_id',
                                    'lens_cat_sys_id', 'image_number'])

        if overwrite_existing is True and os.path.exists(out_file):
            os.remove(out_file)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_df.to_sql('lensed_agn', con=engine)

        if return_df is True:
            return agn_df

    def output_lensed_sne_truth(self, matched_hosts, matched_lenses,
                                matched_sys_cat, out_file,
                                return_df=True, id_offset=0,
                                overwrite_existing=False):

        """
        Create the final properly formatted lens galaxy truth catalog.

        Parameters
        ----------
        matched_lenses: pandas dataframe
            Dataframe of matched DC2 lenses

        matched_sys_cat: pandas dataframe
            Dataframe of matched lens catalog systems

        truth_cat_filename: str
            Filename for sqlite truth catalog

        return_df: bool, default=True
            Option to return dataframe after saving to sqlite file

        id_offset: int, default=0
            Add offset to `dc2_sys_id` parameter.

        overwrite_existing: bool, default=False

        Returns
        -------
        lens_df: pandas dataframe
            Return the pandas dataframe format of the truth catalog.
        """

        new_entries = []

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):

                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id_num = i
                image_number = j
                new_sys_id = 'GLSNE_%i' % new_sys_id_num
                gal_unique_id = 'GLSNE_sne_%i_%i' % (new_sys_id_num, image_number)

                ra_lens = matched_lenses.iloc[i]['ra']
                dec_lens = matched_lenses.iloc[i]['dec']
                id_lens = matched_lenses.iloc[i]['galaxy_id']
                sn_x = matched_sys_cat.iloc[i]['snx']
                sn_y = matched_sys_cat.iloc[i]['sny']
                img_x = matched_sys_cat.iloc[i]['x_img'][j]
                img_y = matched_sys_cat.iloc[i]['y_img'][j]
                delta_ra = matched_sys_cat.iloc[i]['x_img'][j] / 3600.0
                delta_dec = matched_sys_cat.iloc[i]['y_img'][j] / 3600.0
                ra = ra_lens + delta_ra/np.cos(np.radians(dec_lens))
                dec = dec_lens + delta_dec

                t0 = matched_sys_cat.iloc[i]['t0']
                t_delay = matched_sys_cat.iloc[i]['t_delay_img'][j]
                mb = matched_sys_cat.iloc[i]['MB']
                mag = matched_sys_cat.iloc[i]['magnification_img'][j]
                x0 = matched_sys_cat.iloc[i]['x0']
                x1 = matched_sys_cat.iloc[i]['x1']
                c = matched_sys_cat.iloc[i]['c']
                host_type = matched_sys_cat.iloc[i]['host_type']

                redshift = matched_sys_cat.iloc[i]['z_src']

                av_mw = matched_lenses.iloc[i]['av_mw']
                rv_mw = matched_lenses.iloc[i]['rv_mw']

                cat_sys_id = matched_sys_cat.iloc[i]['system_id']

                new_row = [gal_unique_id, gal_unique_id, ra, dec, sn_x, sn_y,
                        img_x, img_y, t0,
                        t_delay, mb, mag, x0, x1, c, host_type,
                        redshift, av_mw, rv_mw,
                        id_lens, new_sys_id, cat_sys_id, image_number]

                new_entries.append(new_row)

        sne_df = pd.DataFrame(new_entries,
                            columns=['unique_id', 'gal_unq_id', 'ra', 'dec', 'x_sne', 'y_sne',
                                    'x_img', 'y_img', 't0', 't_delay', 'MB', 'magnification',
                                    'x0', 'x1', 'c', 'host_type', 'redshift',
                                    'av_mw', 'rv_mw', 'lens_id',
                                    'dc2_sys_id', 'lens_cat_sys_id',
                                    'image_number'])

        if overwrite_existing is True and os.path.exists(out_file):
            os.remove(out_file)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        sne_df.to_sql('lensed_sne', con=engine)

        if return_df is True:
            return sne_df