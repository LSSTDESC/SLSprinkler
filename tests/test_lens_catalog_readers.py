import sys
sys.path.append('..')
import unittest
import pandas as pd
from sprinkler import OM10Reader

class testOM10Reader(unittest.TestCase):

    def test_load_reader(self):

        om10_reader = OM10Reader('../data/twinkles_lenses_cosmoDC2_v1.1.4.fits')
        lensed_agn_cat = om10_reader.load_catalog()
        print(pd.DataFrame(lensed_agn_cat))

if __name__ == '__main__':
    unittest.main()