import unittest, ephem, random
import aipy as a, numpy as n

DIM = 128

class TestClean(unittest.TestCase):
    def test_clean2dc(self):
        res = n.zeros((DIM,DIM), dtype=n.complex64)
        ker = n.zeros((DIM,DIM), dtype=n.complex64)
        mdl = n.zeros((DIM,DIM), dtype=n.complex64)
        area = n.zeros((DIM,DIM), dtype=n.int64)
        self.assertRaises(ValueError, a._deconv.clean, \
            res,ker,mdl,area.astype(n.float))
        ker[0,0] = 1.
	res[2,2] = 1.
        res[1,1] = 2.; res[0,0] = 1.; res[5,5] = 1.
        area[:4][:4] = 1
        rv = a._deconvGPU.clean(res,ker,mdl,area,tol=1e-8)
        print rv
        print res
        self.assertAlmostEqual(res[0,0], 0, 3)
        self.assertEqual(res[5,5], 1)
    

if __name__ == '__main__':
    unittest.main()
