import pandas as pd
import numpy as np

__all__ = ['Sprinkler']

class Sprinkler():

    def __init__(self, 
                 gl_agn_cat, glagn_reader,
                 gl_sne_cat, glsne_reader,
                 gal_cat, gal_reader,
                 avoid_gal_ids=None):

        self.gl_agn_cat = pd.DataFrame(glagn_reader(gl_agn_cat).load_catalog())
        self.gl_sne_cat = pd.DataFrame(glsne_reader(gl_sne_cat).load_catalog())
        self.gal_cat = pd.DataFrame(gal_reader(gal_cat).load_catalog())

    def match_agn(self, galz, gal_mag):

        return

    def sprinkle(self):

        return