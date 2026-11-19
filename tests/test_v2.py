import unittest
import numpy as np
from v2.preprocessing import transform

class TestV2(unittest.TestCase):
    def test_missing_hands(self):
        x = transform(np.zeros((15,84)))
        self.assertEqual(x.shape, (15,88))
        self.assertFalse(np.any(x))

    def test_translation_invariance(self):
        x = np.zeros((15,2,21,2), dtype=np.float32)
        x[:,0,:,:] = [100,100]
        x[:,0,9,:] = [100,120]
        x[:,1,:,:] = [200,200]
        x[:,1,9,:] = [200,220]
        y = x + np.array([30,40],dtype=np.float32)
        self.assertTrue(np.allclose(transform(x.reshape(15,84)), transform(y.reshape(15,84))))

if __name__ == '__main__': unittest.main()
