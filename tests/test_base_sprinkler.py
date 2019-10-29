import sys
sys.path.append('..')
import unittest
from sprinkler import OM10Reader, GoldsteinSNeCatReader, DC2Reader
from sprinkler import BaseSprinkler

class testBaseSprinkler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.gl_agn_cat = '../data/twinkles_lenses_cosmoDC2_v1.1.4.fits'
        cls.gl_sne_cat = '../data/glsne_test.h5'
        cls.gal_cat = '../data/test_ddf.pkl'

        cls.glagn_reader = OM10Reader
        cls.glsne_reader = GoldsteinSNeCatReader
        cls.gal_reader = DC2Reader

        cls.sl_sprinkler = BaseSprinkler(cls.gl_agn_cat, cls.glagn_reader,
                                         cls.gl_sne_cat, cls.glsne_reader,
                                         cls.gal_cat, cls.gal_reader)

    def test_loader(self):

        print(self.sl_sprinkler.gl_agn_cat)
        print(self.sl_sprinkler.gl_sne_cat)
        print(self.sl_sprinkler.gal_cat)

    def test_match_agn(self):

        self.sl_sprinkler.sprinkle_agn()

    def test_match_sne(self):

        self.sl_sprinkler.sprinkle_sne()


if __name__ == '__main__':
    unittest.main()
