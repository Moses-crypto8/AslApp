import unittest
import numpy as np
import importlib.util
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch not installed")
import torch
from v2.gru_model import SignGRU
from v2.temporal_dataset import split_indices

class GRUTests(unittest.TestCase):
    def test_forward(self):
        m = SignGRU(num_classes=4)
        self.assertEqual(tuple(m(torch.zeros(3,15,88)).shape), (3,4))
    def test_bad_features(self):
        with self.assertRaises(ValueError): SignGRU()(torch.zeros(2,15,84))
    def test_group_split(self):
        labels=np.array([0,1]*20)
        groups=np.repeat(np.arange(10),4)
        a,b=split_indices(labels, groups=groups)
        self.assertFalse(set(groups[a]) & set(groups[b]))
if __name__ == '__main__': unittest.main()
