import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from normalization import normalize_sequence,relocate_sequence
from database import load_database
from features import feature_columns
from trainer import split_data,load_model
from inference import Predictor
ROOT=Path(__file__).resolve().parents[1]

class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_database(ROOT/'sign_language.csv')
        cls.raw=cls.df[feature_columns()].to_numpy().reshape(-1,15,84)

    def test_random_translation_and_uniform_scale_invariance(self):
        rng=np.random.default_rng(21)
        for seq in self.raw[::20]:
            expected=normalize_sequence(seq)
            for _ in range(8):
                transformed=relocate_sequence(seq,rng.uniform(.2,3),rng.uniform(-1000,1000,2))
                np.testing.assert_allclose(normalize_sequence(transformed),expected,rtol=1e-10,atol=1e-10)

    def test_motion_path_is_retained(self):
        hands=np.zeros((15,2,21,2))
        for f in range(15):
            hands[f,0]=[100+f*5,200]
            hands[f,0,9]=[100+f*5,220]
        out=normalize_sequence(hands.reshape(15,84))
        self.assertAlmostEqual(out[-1,0]-out[0,0],3.5)
        np.testing.assert_allclose(out[:,84:168],np.repeat(out[0:1,84:168],15,axis=0))
        self.assertTrue(np.all(out[:,-2]==1));self.assertTrue(np.all(out[:,-1]==0))
        self.assertFalse(np.any(out[:,42:84]))

    def test_relative_hand_position_is_retained(self):
        hands=np.zeros((15,2,21,2))
        hands[:,0]=[100,100];hands[:,0,9]=[100,120]
        hands[:,1]=[200,100];hands[:,1,9]=[200,120]
        out=normalize_sequence(hands.reshape(15,84))
        np.testing.assert_allclose(out[:,42]-out[:,0],5)

    def test_full_holdout_prediction_consistency(self):
        _,ids=split_data(self.df)
        model=load_model(ROOT/'models/asl_model.joblib')['model']
        raw=self.raw[ids]
        base=model.predict(pd.DataFrame(raw.reshape(-1,1260),columns=feature_columns()))
        for scale,offset in [(.5,(500,300)),(1.,(-250,-150)),(1.6,(-300,100)),(2.,(0,0))]:
            moved=np.stack([relocate_sequence(s,scale,offset) for s in raw])
            got=model.predict(pd.DataFrame(moved.reshape(-1,1260),columns=feature_columns()))
            np.testing.assert_array_equal(base,got)
        p=Predictor(ROOT/'models/asl_model.joblib')
        self.assertEqual(p.predict(raw[0])[0],p.predict(relocate_sequence(raw[0],.6,(100,300)))[0])

    def test_degenerate_and_empty_inputs(self):
        self.assertFalse(np.any(normalize_sequence(np.zeros((15,84)))))
        with self.assertRaises(ValueError):normalize_sequence(np.ones((15,84)))
        with self.assertRaises(ValueError):normalize_sequence(np.zeros((10,84)))

if __name__=='__main__':unittest.main()
