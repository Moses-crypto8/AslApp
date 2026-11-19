import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from database import load_database, append_update
from features import feature_columns, sequence_to_row
from inference import Predictor, SequenceBuffer
from trainer import load_model, evaluate_payload, split_data

ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_database(ROOT/'sign_language.csv')

    def test_original_accuracy_and_disjoint_holdout(self):
        r=evaluate_payload(load_model(ROOT/'models/asl_pixel_baseline.joblib'),self.df,ROOT/'sign_language.csv')
        self.assertEqual(r['correct'],115)
        self.assertEqual(r['test_samples'],118)
        self.assertEqual(r['identical_features_across_split'],0)
        self.assertFalse(set(r['train_indices']) & set(r['test_indices']))

    def test_feature_roundtrip(self):
        x=self.df.iloc[0][feature_columns()].to_numpy().reshape(15,84)
        row=sequence_to_row(x,'example')
        np.testing.assert_equal([row[k] for k in feature_columns()],x.flatten())

    def test_gap_clears_sequence(self):
        b=SequenceBuffer()
        for _ in range(14):self.assertIsNone(b.push([1]*84))
        self.assertIsNone(b.push(None))
        for _ in range(14):self.assertIsNone(b.push([2]*84))
        self.assertEqual(len(b.push([2]*84)),15)
        self.assertEqual(b.frames,[])

    def test_inference_matches_saved_model(self):
        p=Predictor(ROOT/'models/asl_model.joblib')
        x=self.df.iloc[0][feature_columns()].to_numpy().reshape(15,84)
        label,confidence=p.predict(x)
        self.assertIn(label,p.classes);self.assertGreaterEqual(confidence,0)
        self.assertEqual(p.predict(np.zeros((15,84))),(None,0.0))
        with self.assertRaises(ValueError):p.predict(np.zeros((14,84)))

    def test_database_update_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/'db.csv';update=Path(d)/'update.csv'
            self.df.iloc[:3].to_csv(db,index=False)
            self.df.iloc[[4]].to_csv(update,index=False)
            append_update(db,update,backup=False)
            self.assertEqual(len(load_database(db)),4)
            before=db.read_bytes()
            bad=self.df.iloc[[4]].copy();bad.iloc[0,0]=np.nan;bad.to_csv(update,index=False)
            with self.assertRaises(ValueError):append_update(db,update,backup=False)
            self.assertEqual(db.read_bytes(),before)

    def test_reordered_schema_canonicalized(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'reordered.csv'
            self.df.iloc[:2,self.df.columns.size-1::-1].to_csv(f,index=False)
            self.assertEqual(list(load_database(f).columns),feature_columns()+['label'])

    def test_insufficient_samples_have_clear_error(self):
        with self.assertRaisesRegex(ValueError,'two terms'):split_data(self.df.iloc[:1])

class CameraTests(unittest.TestCase):
    def test_real_detector_on_blank_image(self):
        from hand_detector2 import handDetector
        d=handDetector()
        try:
            image=np.zeros((480,640,3),dtype=np.uint8)
            d.find_hands(image,draw=False)
            self.assertEqual(d.find_position(image,draw=False),[])
        finally:d.hands.close()

    def test_camera_failure_releases_resources(self):
        from camera import open_camera
        capture=MagicMock();capture.isOpened.return_value=False
        with patch('camera.cv2.VideoCapture',return_value=capture),patch('camera.cv2.destroyAllWindows'):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                with open_camera(99):pass
        capture.release.assert_called_once()

    def test_recognizer_camera_loop(self):
        from contextlib import contextmanager
        import recognizer
        detector=MagicMock();detector.find_position.return_value=[]
        @contextmanager
        def fake_camera(index):yield MagicMock(),detector
        frame=np.zeros((480,640,3),dtype=np.uint8)
        with patch.object(recognizer,'open_camera',fake_camera),patch.object(recognizer,'read_frame',return_value=frame),patch.object(recognizer.cv2,'imshow'),patch.object(recognizer.cv2,'waitKey',return_value=ord('q')):
            recognizer.run_recognizer(str(ROOT/'models/asl_model.joblib'))

    def test_collector_sequence_and_csv(self):
        from contextlib import contextmanager
        import collector
        detector=MagicMock()
        detector.find_position.return_value=[('Left',[[i,100+i,200+i] for i in range(21)])]
        @contextmanager
        def fake_camera(index):yield MagicMock(),detector
        frame=np.zeros((480,640,3),dtype=np.uint8)
        detector.find_position.side_effect=[detector.find_position.return_value]*15+[[]]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'capture.csv'
            with patch.object(collector,'open_camera',fake_camera),patch.object(collector,'read_frame',return_value=frame),patch.object(collector.cv2,'imshow'),patch.object(collector.cv2,'getWindowProperty',return_value=1),patch.object(collector.time,'monotonic',side_effect=[i*.05 for i in range(16)]),patch.object(collector.cv2,'waitKey',side_effect=[32]+[-1]*14+[13]):
                collector.collect_samples('new sign',out,target_samples=1)
            result=load_database(out)
            self.assertEqual(result.shape,(1,1261));self.assertEqual(result.label.iloc[0],'new sign')

@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch not installed")
class TemporalIntegration(unittest.TestCase):
    def test_checkpoint_inference_reproduces_report(self):
        from v2.preprocessing import transform
        import torch
        df=load_database(ROOT/'sign_language.csv'); _,ids=split_data(df)
        p=Predictor(ROOT/'v2_runs/gru_baseline/best_gru.pt','gru')
        seq=df.iloc[ids][feature_columns()].to_numpy(dtype=np.float32).reshape(-1,15,84)
        with torch.inference_mode():
            y=p.model(torch.from_numpy(np.stack([transform(s) for s in seq]))).argmax(-1).numpy()
        predicted=np.array(p.classes)[y]
        self.assertEqual(int(np.sum(predicted==df.iloc[ids].label.to_numpy())),114)
        label,confidence=p.predict(seq[0]);self.assertIn(label,p.classes);self.assertGreater(confidence,0)

if __name__=='__main__':unittest.main()
