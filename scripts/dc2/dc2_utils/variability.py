from builtins import range
from builtins import object
import numpy as np
import linecache
import math
import os
import gzip
import numbers
import multiprocessing
import json as json
import numpy as np
from lsst.sims.catalogs.decorators import register_method, compound
from lsst.sims.catUtils.mixins import Variability

__all__ = ["ExtraGalacticVariabilityModels", "VariabilityAGN"]

class ExtraGalacticVariabilityModels(Variability):
    """
    A mixin providing the model for AGN variability.
    """

    _agn_walk_start_date = 58350.0
    _agn_threads = 1
    # Make the following class attribute a list so that we can modify
    # it globally for all subclasses from the top-level code.
    filters_to_simulate = ['u', 'g', 'r', 'i', 'z', 'y']

    @register_method('applyAgn')
    def applyAgn(self, valid_dexes, params, expmjd,
                 variability_cache=None, redshift=None):

        if redshift is None:
            redshift_arr = self.column_by_name('redshift')
        else:
            redshift_arr = redshift

        if len(params) == 0:
            return np.array([[],[],[],[],[],[]])

        #if isinstance(expmjd, numbers.Number):
        dMags = np.zeros((6, self.num_variable_obj(params)))
        #    max_mjd = expmjd
        #    min_mjd = expmjd
        #    mjd_is_number = True
        #else:
        #    dMags = np.zeros((6, self.num_variable_obj(params), len(expmjd)))
        max_mjd = max(expmjd)
        min_mjd = min(expmjd)
        #    mjd_is_number = False

        assert len(expmjd) == len(valid_dexes[0]), \
            "Length of MJD list (%i) and # of AGN (%i) not equal" % (len(expmjd), len(valid_dexes[0]))

        seed_arr = params['seed']

        duration_observer_frame = max_mjd - self._agn_walk_start_date

        if duration_observer_frame < 0 or min_mjd < self._agn_walk_start_date:
            raise RuntimeError("WARNING: Time offset greater than minimum epoch.  " +
                               "Not applying variability. "+
                               "expmjd: %e should be > start_date: %e  " % (min_mjd, self._agn_walk_start_date) +
                               "in applyAgn variability method")

        if self._agn_threads == 1 or len(valid_dexes[0])==1:
            for filt_num, filt_name in list(enumerate(['u', 'g', 'r', 'i', 'z', 'y'])):
                if filt_name not in self.filters_to_simulate:
                    continue
                tau_arr = params['agn_tau_%s' % filt_name].astype(float)
                sf_arr = params['agn_sf_%s' % filt_name].astype(float)
                for i_obj in valid_dexes[0]:
                    seed = seed_arr[i_obj]
                    tau_filt = tau_arr[i_obj]
                    time_dilation = 1.0+redshift_arr[i_obj]
                    sf_filt = sf_arr[i_obj]
                    obj_mjd = expmjd[i_obj]
                    dMags[filt_num][i_obj] = self._simulate_agn(obj_mjd, tau_filt, time_dilation, sf_filt, seed)
        else:
            p_list = []

            mgr = multiprocessing.Manager()
            #if mjd_is_number:
            out_struct = mgr.Array('d', [0]*len(valid_dexes[0]))
            #else:
            #    out_struct = mgr.dict()

            #################
            # Try to subdivide the AGN into batches such that the number
            # of time steps simulated by each thread is close to equal

            for filt_num, filt_name in list(enumerate(['u', 'g', 'r', 'i', 'z', 'y'])):
                if filt_name not in self.filters_to_simulate:
                    continue
                tot_steps = 0
                n_steps = []
                tau_arr = params['agn_tau_%s' % filt_name].astype(float)
                sf_arr = params['agn_sf_%s' % filt_name].astype(float)

                for tt, zz in zip(tau_arr[valid_dexes], redshift_arr[valid_dexes]):
                    dilation = 1.0+zz
                    dt = tt/100.0
                    dur = (duration_observer_frame/dilation)
                    nt = dur/dt
                    tot_steps += nt
                    n_steps.append(nt)

                batch_target = tot_steps/self._agn_threads
                i_start_arr = [0]
                i_end_arr = []
                current_batch = n_steps[0]
                for ii in range(1,len(n_steps),1):
                    current_batch += n_steps[ii]
                    if ii == len(n_steps)-1:
                        i_end_arr.append(len(n_steps))
                    elif len(i_start_arr)<self._agn_threads:
                        if current_batch>=batch_target:
                            i_end_arr.append(ii)
                            i_start_arr.append(ii)
                            current_batch = n_steps[ii]

                if len(i_start_arr) != len(i_end_arr):
                    raise RuntimeError('len i_start %d len i_end %d; dexes %d' %
                                    (len(i_start_arr),
                                        len(i_end_arr),
                                        len(valid_dexes[0])))
                assert len(i_start_arr) <= self._agn_threads
                ############

                # Actually simulate the AGN on the the number of threads allotted
                for i_start, i_end in zip(i_start_arr, i_end_arr):
                    dexes = valid_dexes[0][i_start:i_end]
                    if mjd_is_number:
                        out_dexes = range(i_start,i_end,1)
                    else:
                        out_dexes = dexes
                    p = multiprocessing.Process(target=self._threaded_simulate_agn,
                                                args=(expmjd[dexes], tau_arr[dexes],
                                                    1.0+redshift_arr[dexes],
                                                    sf_arr[dexes],
                                                    seed_arr[dexes],
                                                    out_dexes,
                                                    out_struct))
                    p.start()
                    p_list.append(p)
                for p in p_list:
                    p.join()

                #if mjd_is_number:
                dMags[filt_num][valid_dexes] = out_struct[:]
                #else:
                #    for i_obj in out_struct.keys():
                #        dMags[filt_num][i_obj] = out_struct[i_obj]

        # for i_filter, filter_name in enumerate(('g', 'r', 'i', 'z', 'y')):
        #     for i_obj in valid_dexes[0]:
        #         dMags[i_filter+1][i_obj] = dMags[0][i_obj]*params['agn_sf%s' % filter_name][i_obj]/params['agn_sfu'][i_obj]

        return dMags

    def _threaded_simulate_agn(self, expmjd, tau_arr,
                               time_dilation_arr, sf_filt_arr,
                               seed_arr, dex_arr, out_struct):

        if isinstance(expmjd, numbers.Number):
            mjd_is_number = True
        else:
            mjd_is_number = False

        for tau, time_dilation, sf_filt, seed, dex in \
        zip(tau_arr, time_dilation_arr, sf_filt_arr, seed_arr, dex_arr):
            out_struct[dex] = self._simulate_agn(expmjd, tau, time_dilation,
                                                 sf_filt, seed)

    def _simulate_agn(self, expmjd, tau, time_dilation, sf_filt, seed):
        """
        Return the delta mag_norm values wrt the infinite-time average
        mag_norm for the provided AGN light curve parameters.  mag_norm is
        the object's un-reddened monochromatic magnitude at 500nm.

        Parameters
        ----------
        expmjd: np.array
            Times at which to evaluate the light curve delta flux values
            in MJD.  Observer frame.
        tau: float
            Variability time scale in days.
        time_dilation: float
            1 + redshift
        sf_filt: float
            Structure function parameter, i.e., asymptotic rms variability on
            long time scales.
        seed: int
            Random number seed.

        Returns
        -------
        np.array of delta mag_norm values.

        Notes
        -----
        This code is based on/stolen from
        https://github.com/astroML/astroML/blob/master/astroML/time_series/generate.py
        """
        expmjd_is_number = isinstance(expmjd, numbers.Number)
        mjds = np.array([expmjd]) if expmjd_is_number else np.array(expmjd)

        if min(mjds) < self._agn_walk_start_date:
            raise RuntimeError(f'mjds must start after {self._agn_walk_start_date}')

        t_obs = np.arange(self._agn_walk_start_date, max(mjds + 1), dtype=float)
        t_rest = t_obs/time_dilation/tau

        rng = np.random.RandomState(seed)
        nbins = len(t_rest)
        steps = rng.normal(0, 1, nbins)
        delta_mag_norm = np.zeros(nbins)
        delta_mag_norm[0] = steps[0]*sf_filt
        for i in range(1, nbins):
            dt = t_rest[i] - t_rest[i - 1]
            delta_mag_norm[i] = (delta_mag_norm[i - 1]*(1. - dt)
                                 + np.sqrt(2*dt)*sf_filt*steps[i])
        dm_out = np.interp(mjds, t_obs, delta_mag_norm)
        return dm_out if not expmjd_is_number else dm_out[0]


class VariabilityAGN(ExtraGalacticVariabilityModels):
    """
    This is a mixin which wraps the methods from the class
    ExtraGalacticVariabilityModels into getters for InstanceCatalogs
    of AGN.  Getters in this method should define columns named like

    delta_columnName

    where columnName is the name of the baseline (non-varying) magnitude
    column to which delta_columnName will be added.  The getters in the
    photometry mixins will know to find these columns and add them to
    columnName, provided that the columns here follow this naming convention.

    Thus: merely including VariabilityStars in the inheritance tree of
    an InstanceCatalog daughter class will activate variability for any column
    for which delta_columnName is defined.
    """

    @compound('delta_lsst_u', 'delta_lsst_g', 'delta_lsst_r',
             'delta_lsst_i', 'delta_lsst_z', 'delta_lsst_y')
    def get_stellar_variability(self):
        """
        Getter for the change in magnitudes due to stellar
        variability.  The PhotometryStars mixin is clever enough
        to automatically add this to the baseline magnitude.
        """

        varParams = self.column_by_name('varParamStr')
        dmag = self.applyVariability(varParams)
        if dmag.shape != (6, len(varParams)):
            raise RuntimeError("applyVariability is returning "
                               "an array of shape %s\n" % dmag.shape
                               + "should be (6, %d)" % len(varParams))
        return dmag
