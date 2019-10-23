import sys
sys.path.append('..')
import unittest
import pandas as pd
from sprinkler import OM10Reader, GoldsteinSNeCatReader, DC2Reader
from sprinkler import Sprinkler

class testSprinkler(unittest.TestCase):

    def test_loader(self):

        gl_agn_cat = '../data/twinkles_lenses_cosmoDC2_v1.1.4.fits'
        gl_sne_cat = '../data/glsne_cosmoDC2_v1.1.4.h5'
        gal_cat = '../data/full_ddf.pkl'

        glagn_reader = OM10Reader
        glsne_reader = GoldsteinSNeCatReader
        gal_reader = DC2Reader


        sl_sprinkler = Sprinkler(gl_agn_cat, glagn_reader,
                                 gl_sne_cat, glsne_reader,
                                 gal_cat, gal_reader)
        print(sl_sprinkler.gl_agn_cat)
        print(sl_sprinkler.gl_sne_cat)
        print(sl_sprinkler.gal_cat)

if __name__ == '__main__':
    unittest.main()