import pandas as pd
import numpy as np
from .base_sprinkler import BaseSprinkler

__all__ = ['DC2Sprinkler']


class DC2Sprinkler(BaseSprinkler):

    def agn_density(self, agn_gal_row):

        return 1.0

    def sne_density(self, sne_gal_row):

        return 1.0
