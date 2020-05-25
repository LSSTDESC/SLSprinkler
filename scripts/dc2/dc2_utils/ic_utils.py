from lsst.sims.utils import _observedFromICRS
from lsst.sims.catUtils.mixins.AstrometryMixin import PhoSimAstrometryBase

__all__ = ['instCatUtils']

class instCatUtils(PhoSimAstrometryBase):

    def get_phosim_coords(self, ra, dec, obs_metadata):

        raObs, decObs = _observedFromICRS(ra, dec, includeRefraction=False,
                                          obs_metadata=obs_metadata,
                                          epoch=2000.0)

        return self._dePrecess(raObs, decObs, obs_metadata)