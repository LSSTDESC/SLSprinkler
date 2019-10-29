import sys
sys.path.append('..')
import unittest
import pandas as pd
from sprinkler import DC2Reader

class testDC2Reader(unittest.TestCase):

    def test_load_reader(self):

        dc2_reader = DC2Reader('../data/test_ddf.pkl')
        dc2_cat = dc2_reader.load_catalog()
        print(pd.DataFrame(dc2_cat))

if __name__ == '__main__':
    unittest.main()