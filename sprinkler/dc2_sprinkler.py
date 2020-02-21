import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from .base_sprinkler import BaseSprinkler

__all__ = ['DC2Sprinkler']


class DC2Sprinkler(BaseSprinkler):

    def __init__(self, agn_density_file_dir, *args):

        super().__init__(*args)
        self.agn_density_file_dir = agn_density_file_dir

    def find_possible_match_agn(self, gal_cat):

        gal_z = gal_cat['redshift'].values
        gal_i_mag = gal_cat['mag_i_agn'].values

        # search the OM10 catalog for all sources +- 0.1 dex in redshift
        # and within .25 mags of the AGN source
        # and within redshift range of cosmoDC2 (z <= 3)
        lens_candidate_idx = []
        for gal_z_on, gal_i_mag_on in zip(gal_z, gal_i_mag):
            w = np.where((np.abs(np.log10(self.gl_agn_cat['z_src']) -
                                 np.log10(gal_z_on)) <= 0.1) &
                         (np.abs(self.gl_agn_cat['mag_i_src'] -
                                 gal_i_mag_on) <= .25))
            lens_candidate_idx.append(w[0])

        return lens_candidate_idx

    def agn_density(self, agn_gal_row):
        
        density_norm = 1.0
        
        bins_z = np.linspace(0.26212831, 3.1, 21)
        dens_z_file = os.path.join(self.agn_density_file_dir,
                                   'agn_z_density.dat')
        dens_z = np.genfromtxt(dens_z_file)
        dens_z_idx = np.digitize(agn_gal_row['redshift'], bins_z)
        density_z = dens_z[dens_z_idx-1]

        bins_mag_i = np.linspace(15.48, 27.22, 21)
        dens_mag_i_file = os.path.join(self.agn_density_file_dir,
                                       'agn_mag_i_density.dat')
        dens_mag_i = np.genfromtxt(dens_mag_i_file)
        dens_mag_i_idx = np.digitize(agn_gal_row['mag_i_agn'], bins_mag_i)
        density_mag_i = dens_mag_i[dens_mag_i_idx-1]
        
        density_val = density_z * density_mag_i * density_norm
        
        return density_val

    def find_possible_match_sne(self, gal_cat):

        gal_z = gal_cat['redshift']
        gal_type = gal_cat['gal_type']

        # search the SNe catalog for all sources +- 0.05 dex in redshift
        # and with matching type
        lens_candidate_idx = []

        w = np.where((np.abs(np.log10(self.gl_sne_cat['z_src']) -
                             np.log10(gal_z)) <= 0.05) &
                     (self.gl_sne_cat['host_type'] == gal_type))
        lens_candidate_idx = w[0]    

        return lens_candidate_idx

    def sne_density(self, sne_gal_row):
        
        density_norm = 0.05
        
        stellar_mass = sne_gal_row['stellar_mass']
        host_type = sne_gal_row['gal_type']
        
        if host_type == 'kinney-elliptical':
            density_host = 0.044 * stellar_mass * 1e-10
        elif host_type == 'kinney-sc':
            density_host = 0.17 * stellar_mass * 1e-10
        elif host_type == 'kinney-starburst':
            density_host = 0.77 * stellar_mass * 1e-10
        
        density_val = density_norm * density_host

        return density_val

    def assign_matches_sne(self, sne_gals, rand_state):
        
        sprinkled_sne_gal_rows = []
        sprinkled_gl_sne_cat_rows = []

        for i in range(len(sne_gals)):
            
            if i % 10000 == 0:
                print(i)
            
            # Draw probability that galaxy is sprinkled
            sne_density = self.sne_density(sne_gals.iloc[i])

            density_draw = rand_state.uniform()
            if density_draw > sne_density: 
                continue
            
            sne_cat_idx = self.find_possible_match_sne(sne_gals.iloc[i])
                
            sne_idx_keep = [x for x in sne_cat_idx
                            if x not in sprinkled_gl_sne_cat_rows]

            if len(sne_idx_keep) == 0:
                continue   

            weight = self.gl_sne_cat['weight'].iloc[sne_idx_keep]
            
            sprinkled_gl_sne_cat_rows.append(
                rand_state.choice(sne_idx_keep, p = weight/np.sum(weight)))
            sprinkled_sne_gal_rows.append(i)
            
        return sprinkled_sne_gal_rows, sprinkled_gl_sne_cat_rows

    def get_new_id(self, gal_id, new_sys_id, image_num):

        """
        Create a new unique id for sprinkled galaxies.
        """

        new_id = ((gal_id+int(1.5e10))*100000 + new_sys_id*8 + image_num)

        return new_id

    def output_lens_galaxy_truth(self, matched_agn_hosts, matched_agn_sys,
                                 matched_sne_hosts, matched_sne_sys, out_file,
                                 return_df=False):

        """
        Output sqlite truth catalogs for foreground lens galaxies for
        lensed AGN and SNe.
        """

        agn_lens_df = self.create_lens_truth_dataframe(matched_agn_hosts,
                                                       matched_agn_sys)
        sne_lens_df = self.create_lens_truth_dataframe(matched_sne_hosts,
                                                       matched_sne_sys, id_offset=2000)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_lens_df.to_sql('agn_lens', con=engine)
        sne_lens_df.to_sql('sne_lens', con=engine)

        if return_df is True:
            return agn_lens_df, sne_lens_df

    def create_lens_truth_dataframe(self, matched_hosts, matched_sys_cat, id_offset=0):

        new_entries = []

        for i in range(len(matched_sys_cat)):
                
            gal_id = matched_hosts.iloc[i]['galaxy_id']
            new_sys_id = i + id_offset
            # The newly inserted lens galaxy keeps the old gal_id
            unique_id = gal_id

            ra_lens = matched_hosts.iloc[i]['ra']
            dec_lens = matched_hosts.iloc[i]['dec']

            # Our lens galaxies are just bulges
            magnorm_lens = matched_sys_cat.iloc[i]['magnorm_lens']
            magnorm_lens_u = magnorm_lens[0]
            magnorm_lens_g = magnorm_lens[1]
            magnorm_lens_r = magnorm_lens[2]
            magnorm_lens_i = magnorm_lens[3]
            magnorm_lens_z = magnorm_lens[4]
            magnorm_lens_y = magnorm_lens[5]
            
            redshift = matched_sys_cat.iloc[i]['z_lens']
            shear_1_dc2 = matched_hosts.iloc[i]['gamma_1']
            shear_2_dc2 = matched_hosts.iloc[i]['gamma_2']
            kappa_dc2 = matched_hosts.iloc[i]['kappa']
            gamma_lenscat = matched_sys_cat.iloc[i]['gamma']
            phi_gamma_lenscat = matched_sys_cat.iloc[i]['phi_gamma']
            shear_1_lenscat = gamma_lenscat * np.cos(2 * phi_gamma_lenscat)
            shear_2_lenscat = gamma_lenscat * np.sin(2 * phi_gamma_lenscat)
            sindex_lens = 4
                
            major_axis_lens = matched_sys_cat.iloc[i]['reff_lens'] / \
                                np.sqrt(1 - matched_sys_cat.iloc[i]['ellip_lens'])
            minor_axis_lens = matched_sys_cat.iloc[i]['reff_lens'] * \
                                np.sqrt(1 - matched_sys_cat.iloc[i]['ellip_lens'])
            position_angle = matched_sys_cat.iloc[i]['phie_lens']*(-1.0)*np.pi/180.0
            ellip_lens = matched_sys_cat.iloc[i]['ellip_lens']
            phie_lens = matched_sys_cat.iloc[i]['phie_lens']
                
            av_internal_lens = matched_sys_cat.iloc[i]['av_lens']
            rv_internal_lens = matched_sys_cat.iloc[i]['rv_lens']
            av_mw = matched_hosts.iloc[i]['av_mw']
            rv_mw = matched_hosts.iloc[i]['rv_mw']
            
            vel_disp_lens = matched_sys_cat.iloc[i]['vel_disp_lens']

            sed_lens = matched_sys_cat.iloc[i]['sed_lens']
                
            cat_sys_id = matched_sys_cat.iloc[i]['system_id']
                
            new_row = [unique_id, ra_lens, dec_lens,
                       magnorm_lens_u, magnorm_lens_g, magnorm_lens_r,
                       magnorm_lens_i, magnorm_lens_z, magnorm_lens_y,
                       redshift, shear_1_dc2, shear_2_dc2, kappa_dc2, 
                       gamma_lenscat, phi_gamma_lenscat, 
                       shear_1_lenscat, shear_2_lenscat, sindex_lens,
                       major_axis_lens, minor_axis_lens,
                       position_angle, ellip_lens, phie_lens, av_internal_lens,
                       rv_internal_lens, av_mw, rv_mw, vel_disp_lens, sed_lens,
                       gal_id, cat_sys_id, new_sys_id]
                
            new_entries.append(new_row)

        lens_df = pd.DataFrame(new_entries, 
                               columns=['unique_id', 'ra_lens', 'dec_lens',
                                        'magnorm_lens_u', 'magnorm_lens_g', 'magnorm_lens_r',
                                        'magnorm_lens_i', 'magnorm_lens_z', 'magnorm_lens_y',
                                        'redshift', 'shear_1_cosmodc2', 'shear_2_cosmodc2',
                                        'kappa_cosmodc2', 'gamma_lenscat', 'phig_lenscat',
                                        'shear_1_lenscat', 'shear_2_lenscat',
                                        'sindex_lens', 'major_axis_lens',
                                        'minor_axis_lens', 'position_angle', 'ellip_lenscat',
                                        'phie_lenscat', 'av_internal_lens', 'rv_internal_lens',
                                        'av_mw', 'rv_mw', 'vel_disp_lenscat',
                                        'sed_lens', 'original_gal_id', 
                                        'lens_cat_sys_id', 'dc2_sys_id'])
        
        return lens_df

    def output_host_galaxy_truth(self, matched_agn_hosts, matched_agn_sys,
                                 matched_sne_hosts, matched_sne_sys, out_file,
                                 return_df=False):

        """
        Output sqlite truth catalogs for host galaxies for
        lensed AGN and SNe.
        """

        agn_host_df = self.create_host_truth_dataframe(matched_agn_hosts,
                                                       matched_agn_sys)
        sne_host_df = self.create_host_truth_dataframe(matched_sne_hosts,
                                                       matched_sne_sys, id_offset=2000)

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_host_df.to_sql('agn_hosts', con=engine)
        sne_host_df.to_sql('sne_hosts', con=engine)

        if return_df is True:
            return agn_host_df, sne_host_df

    def create_host_truth_dataframe(self, matched_hosts, matched_sys_cat,
                                    id_offset=0):

        new_entries = []

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):
                
                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id = i + id_offset
                image_number = j
                unique_id = self.get_new_id(gal_id, new_sys_id, image_number)

                ra_lens = matched_hosts.iloc[i]['ra']
                dec_lens = matched_hosts.iloc[i]['dec']
                
                x_src = matched_sys_cat.iloc[i]['x_src']
                y_src = matched_sys_cat.iloc[i]['y_src']
                x_img = matched_sys_cat.iloc[i]['x_img'][j]
                y_img = matched_sys_cat.iloc[i]['y_img'][j]
                
                delta_ra = np.radians(matched_sys_cat.iloc[i]['x_img'][j] / 3600.0)
                delta_dec = np.radians(matched_sys_cat.iloc[i]['y_img'][j] / 3600.0)
                ra_host = ra_lens + delta_ra/np.cos(dec_lens)
                dec_host = dec_lens + delta_dec
                
                mag_adjust = 2.5*np.log10(np.abs(matched_sys_cat.iloc[i]['magnification_img'][j]))
                magnorm_disk = matched_hosts.iloc[i]['magnorm_disk'] - mag_adjust
                magnorm_disk_u = magnorm_disk[0]
                magnorm_disk_g = magnorm_disk[1]
                magnorm_disk_r = magnorm_disk[2]
                magnorm_disk_i = magnorm_disk[3]
                magnorm_disk_z = magnorm_disk[4]
                magnorm_disk_y = magnorm_disk[5]
                magnorm_bulge = matched_hosts.iloc[i]['magnorm_bulge'] - mag_adjust
                magnorm_bulge_u = magnorm_bulge[0]
                magnorm_bulge_g = magnorm_bulge[1]
                magnorm_bulge_r = magnorm_bulge[2]
                magnorm_bulge_i = magnorm_bulge[3]
                magnorm_bulge_z = magnorm_bulge[4]
                magnorm_bulge_y = magnorm_bulge[5]
                redshift = matched_sys_cat.iloc[i]['z_src']
                shear_1 = 0. #matched_hosts.iloc[i]['gamma_1']
                shear_2 = 0. #matched_hosts.iloc[i]['gamma_2']
                kappa = 0. #matched_hosts.iloc[i]['kappa']
                sindex_bulge = 4
                sindex_disk = 1
                
                major_axis_disk = matched_hosts.iloc[i]['semi_major_axis_disk']
                major_axis_bulge = matched_hosts.iloc[i]['semi_major_axis_bulge']
                minor_axis_disk = matched_hosts.iloc[i]['semi_minor_axis_disk']
                minor_axis_bulge = matched_hosts.iloc[i]['semi_minor_axis_bulge']
                position_angle = matched_hosts.iloc[i]['position_angle']
                
                av_internal_disk = matched_hosts.iloc[i]['av_internal_disk']
                av_internal_bulge = matched_hosts.iloc[i]['av_internal_bulge']
                rv_internal_disk = matched_hosts.iloc[i]['rv_internal_disk']
                rv_internal_bulge = matched_hosts.iloc[i]['rv_internal_bulge']
                av_mw = matched_hosts.iloc[i]['av_mw']
                rv_mw = matched_hosts.iloc[i]['rv_mw']
                
                sed_disk_host = matched_hosts.iloc[i]['sed_disk']
                sed_bulge_host = matched_hosts.iloc[i]['sed_bulge']
                
                cat_sys_id = matched_sys_cat.iloc[i]['system_id']
                
                new_row = [unique_id, x_src, y_src, x_img, y_img,
                           ra_lens, dec_lens, ra_host, dec_host,
                           magnorm_disk_u, magnorm_disk_g, magnorm_disk_r,
                           magnorm_disk_i, magnorm_disk_z, magnorm_disk_y,
                           magnorm_bulge_u, magnorm_bulge_g, magnorm_bulge_r,
                           magnorm_bulge_i, magnorm_bulge_z, magnorm_bulge_y,
                           redshift, shear_1, shear_2, kappa, sindex_bulge, sindex_disk,
                           major_axis_disk, major_axis_bulge, minor_axis_disk, minor_axis_bulge,
                           position_angle, av_internal_disk, av_internal_bulge, rv_internal_disk,
                           rv_internal_bulge, av_mw, rv_mw, sed_disk_host, sed_bulge_host,
                           gal_id, cat_sys_id, new_sys_id, image_number]
                
                new_entries.append(new_row)

        host_df = pd.DataFrame(new_entries, 
                               columns=['unique_id', 'x_src', 'y_src', 'x_img', 'y_img',
                                        'ra_lens', 'dec_lens', 'ra_host', 'dec_host',
                                        'magnorm_disk_u', 'magnorm_disk_g', 'magnorm_disk_r',
                                        'magnorm_disk_i', 'magnorm_disk_z', 'magnorm_disk_y',
                                        'magnorm_bulge_u', 'magnorm_bulge_g', 'magnorm_bulge_r',
                                        'magnorm_bulge_i', 'magnorm_bulge_z', 'magnorm_bulge_y',
                                        'redshift', 'shear_1', 'shear_2',
                                        'kappa', 'sindex_bulge', 'sindex_disk',
                                        'major_axis_disk', 'major_axis_bulge', 'minor_axis_disk',
                                        'minor_axis_bulge', 'position_angle',
                                        'av_internal_disk', 'av_internal_bulge',
                                        'rv_internal_disk', 'rv_internal_bulge',
                                        'av_mw', 'rv_mw', 'sed_disk_host',
                                        'sed_bulge_host', 'original_gal_id', 
                                        'lens_cat_sys_id', 'dc2_sys_id', 'image_number'])
        
        return host_df

    def output_lensed_agn_truth(self, matched_hosts,
                                matched_sys_cat, out_file,
                                return_df=False, id_offset=0):

        new_entries = []

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):

                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id = i + id_offset
                image_number = j
                gal_unique_id = self.get_new_id(gal_id, new_sys_id, image_number)

                ra_lens = matched_hosts.iloc[i]['ra']
                dec_lens = matched_hosts.iloc[i]['dec']
                delta_ra = np.radians(matched_sys_cat.iloc[i]['x_img'][j] / 3600.0)
                delta_dec = np.radians(matched_sys_cat.iloc[i]['y_img'][j] / 3600.0)
                ra = ra_lens + delta_ra/np.cos(dec_lens)
                dec = dec_lens + delta_dec

                redshift = matched_sys_cat.iloc[i]['z_src']
                t_delay = matched_sys_cat.iloc[i]['t_delay_img'][j]

                magnorm = matched_hosts.iloc[i]['magnorm_agn']
                mag = matched_sys_cat.iloc[i]['magnification_img'][j]

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

                cat_sys_id = matched_sys_cat.iloc[i]['system_id']

                new_row = [gal_unique_id, gal_unique_id, ra, dec, 
                           redshift, t_delay, magnorm, mag, 
                           seed, agn_tau_u, agn_tau_u, agn_tau_u,
                           agn_tau_u, agn_tau_u, agn_tau_u,
                           agn_sf_u, agn_sf_g, agn_sf_r, 
                           agn_sf_i, agn_sf_z, agn_sf_y, new_sys_id,
                           cat_sys_id, image_number]
                
                new_entries.append(new_row)

        agn_df = pd.DataFrame(new_entries,
                              columns=['unique_id', 'gal_unq_id', 'ra', 'dec',
                                       'redshift', 't_delay', 'magnorm', 'magnification',
                                       'seed', 'agn_tau_u', 'agn_tau_g', 'agn_tau_r',
                                       'agn_tau_i', 'agn_tau_z', 'agn_tau_y',
                                       'agn_sf_u', 'agn_sf_g', 'agn_sf_r',
                                       'agn_sf_i', 'agn_sf_z', 'agn_sf_y', 'dc2_sys_id',
                                       'lens_cat_sys_id', 'image_number'])

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        agn_df.to_sql('lensed_agn', con=engine)

        if return_df is True:
            return agn_df

    def output_lensed_sne_truth(self, matched_hosts,
                                matched_sys_cat, out_file,
                                return_df=False, id_offset=0):

        new_entries = []

        for i in range(len(matched_sys_cat)):
            for j in range(matched_sys_cat.iloc[i]['n_img']):

                gal_id = matched_hosts.iloc[i]['galaxy_id']+j
                new_sys_id = i + id_offset
                image_number = j
                gal_unique_id = self.get_new_id(gal_id, new_sys_id, image_number)

                ra_lens = matched_hosts.iloc[i]['ra']
                dec_lens = matched_hosts.iloc[i]['dec']
                delta_ra = np.radians(matched_sys_cat.iloc[i]['x_img'][j] / 3600.0)
                delta_dec = np.radians(matched_sys_cat.iloc[i]['y_img'][j] / 3600.0)
                ra = ra_lens + delta_ra/np.cos(dec_lens)
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

                cat_sys_id = matched_sys_cat.iloc[i]['system_id']

                new_row = [gal_unique_id, gal_unique_id, ra, dec, t0,
                           t_delay, mb, mag, x0, x1, c, host_type,
                           redshift, new_sys_id, cat_sys_id, image_number]
                
                new_entries.append(new_row)

        sne_df = pd.DataFrame(new_entries,
                              columns=['unique_id', 'gal_unq_id', 'ra', 'dec',
                                       't0', 't_delay', 'MB', 'magnification',
                                       'x0', 'x1', 'c', 'host_type', 'redshift',
                                       'dc2_sys_id', 'lens_cat_sys_id', 
                                       'image_number'])

        engine = create_engine('sqlite:///%s' % out_file, echo=False)
        sne_df.to_sql('lensed_sne', con=engine)

        if return_df is True:
            return sne_df

    def generate_matched_catalogs(self):

        agn_hosts, agn_systems = self.sprinkle_agn()
        sne_hosts, sne_systems = self.sprinkle_sne()

        return agn_hosts, agn_systems, sne_hosts, sne_systems
