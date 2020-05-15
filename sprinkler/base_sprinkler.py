import numpy as np

__all__ = ['BaseSprinkler']


class BaseSprinkler():

    def calc_mu_e(self, app_mag, radius_arcsec, redshift):

        """
        Calculate the mu_e parameter using Eq. 7 in Hyde and Bernardi 2009

        Parameters
        ----------

        app_mag: float
        Apparent observed LSST r-band magnitude

        radius_arcsec: float
        Galaxy radius in arcsec. We use the bulge radius since we want to look at large elliptical galaxies.

        redshift: float
        Galaxy redshift

        Returns
        -------

        mu_e: float
        Mu_e parameter for Fundamental Plane
        """

        mu_e = app_mag + 5*np.log10(radius_arcsec) + 2.5*np.log10(2*np.pi) - \
            10*np.log10(1+redshift)

        return mu_e

    def calc_velocity_dispersion(self, radius, mu_e):

        """
        Calculate velocity dispersion using Fundamental Plane relation in
        Equation 6 of Hyde and Bernardi 2009.

        a, b, c coefficient values taken from Table 2 in same paper,
        orthogonal for the r-band

        We include the intrinsic error for the r-band orthogonal values
        from the table with an rms of 0.0578

        Parameters
        ----------

        radius: float
        Galaxy radius in kpc.

        mu_e: float
        Mu_e parameter calculated using `calc_mu_e` function

        Returns
        -------

        sigma_fp: float
        Velocity Dispersion in km/s for galaxies
        """

        a = 1.4335
        b = 0.3150
        c = -8.8979

        rand_state = np.random.RandomState(seed=88)

        log_sigma_fp = (np.log10(radius) - b*mu_e - c)/a
        orthog_r_err = rand_state.normal(scale=0.0578, size=len(log_sigma_fp))
        log_sigma_fp += orthog_r_err

        return np.power(10, log_sigma_fp)
