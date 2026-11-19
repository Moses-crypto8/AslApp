import unittest
from pathlib import Path
import tempfile
import numpy as np
from segmentation import SignCapture,Segment,resample_segment,motion_rate


def hand(x=100):
    a=np.zeros((2,21,2))
    a[0]=[x,100];a[0,9]=[x,120]
    return a.reshape(84)

class SegmentTests(unittest.TestCase):
    def test_manual_complete_phrase_not_15_frame_chunks(self):
        c=SignCapture()
        self.assertIsNone(c.update(hand(),0,toggle=True))
        for i in range(1,90):self.assertIsNone(c.update(hand(100+i),i/30))
        s=c.update(hand(190),3,toggle=True)
        self.assertEqual(len(s.frames),91);self.assertEqual(s.duration,3)
        self.assertFalse(c.active)
        got=resample_segment(s.frames,s.times)
        self.assertEqual(got.shape,(15,84));self.assertEqual(got[-1,0],190)

    def test_manual_static_sign_can_finish(self):
        c=SignCapture()
        c.update(hand(),0,toggle=True)
        for i in range(1,21):s=c.update(hand(),i*.05,toggle=i==20)
        self.assertIsNotNone(s);self.assertAlmostEqual(s.duration,1)

    def test_automatic_static_arrival_finishes_once(self):
        c=SignCapture(mode='auto')
        result=[]
        for i in range(60):
            s=c.update(hand(),i*.05)
            if s:result.append(s)
        self.assertEqual(len(result),1)
        self.assertGreaterEqual(result[0].duration,.35)
        self.assertFalse(c.active)

    def test_auto_movement_then_pause(self):
        c=SignCapture(mode='auto');found=[]
        for i in range(80):
            x=100+min(i,30)*2
            s=c.update(hand(x),i*.05)
            if s:found.append(s)
        self.assertEqual(len(found),1)
        self.assertGreater(found[0].duration,1.4)
        self.assertEqual(found[0].reason,'pause')

    def test_long_tracking_gap_discards(self):
        c=SignCapture(mode='auto',max_seconds=.6);c.update(hand(),0,toggle=True)
        c.update(hand(),.1);c.update(None,.2);self.assertIsNone(c.update(None,.7))
        self.assertFalse(c.active);self.assertIn('discarded',c.status)

    def test_return_after_long_gap_does_not_stitch(self):
        c=SignCapture(mode='auto');c.update(hand(),0,toggle=True)
        c.update(None,.1);self.assertIsNone(c.update(hand(),.5))
        self.assertFalse(c.active)

    def test_small_tracking_gap_tolerated(self):
        c=SignCapture();c.update(hand(),0,toggle=True)
        c.update(hand(),.1);c.update(None,.2);c.update(hand(),.3)
        s=c.update(hand(),.5,toggle=True)
        self.assertIsNotNone(s);self.assertEqual(len(s.frames),4)

    def test_timeout_and_cancel(self):
        c=SignCapture(max_seconds=1)
        c.update(hand(),0,toggle=True)
        for i in range(1,11):s=c.update(hand(),i*.1)
        self.assertEqual(s.reason,'maximum duration')
        c.update(hand(),2,toggle=True);c.update(hand(),2.1,cancel=True)
        self.assertFalse(c.active);self.assertEqual(c.frames,[])

    def test_too_short_rejected(self):
        c=SignCapture();c.update(hand(),0,toggle=True)
        c.update(hand(),.1);self.assertIsNone(c.update(hand(),.2,toggle=True))
        self.assertFalse(c.active);self.assertIn('Too short',c.status)

    def test_temporal_sampling_uses_elapsed_time(self):
        frames=np.array([hand(100),hand(110),hand(200)])
        out=resample_segment(frames,[0,.1,1],count=3)
        np.testing.assert_allclose(out[:,0],[100,150,200])
        self.assertFalse(np.any(out[:,42:]))

    def test_missing_hand_not_interpolated_into_fake_hand(self):
        a=hand();b=np.zeros(84)
        out=resample_segment([a,b],[0,1],count=5)
        np.testing.assert_equal(out[-1],b)
        self.assertTrue(np.array_equal(out[3],b))
        self.assertTrue(np.array_equal(out[1],a))

    def test_invalid_timing_and_settings(self):
        with self.assertRaises(ValueError):SignCapture(max_seconds=.1)
        with self.assertRaises(ValueError):SignCapture(idle_seconds=0)
        with self.assertRaises(ValueError):resample_segment([hand(),hand()],[1,1])
        c=SignCapture();c.update(None,1)
        with self.assertRaises(ValueError):c.update(None,.5)

    def test_raw_archive_keeps_duration_and_all_frames(self):
        from collector import save_recordings
        from database import load_database
        times=np.linspace(0,3,91)
        frames=[hand(100+i) for i in range(91)]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'phrase.csv'
            save_recordings([Segment(frames,times.tolist(),'manual')],'phrase',out,'manual')
            df=load_database(out);self.assertEqual(df.shape,(1,1261))
            with np.load(out.with_suffix('.recordings.npz'),allow_pickle=False) as raw:
                self.assertEqual(raw['frames'].shape,(91,84));self.assertAlmostEqual(raw['times'][-1],3)
                np.testing.assert_array_equal(raw['offsets'],[0,91])
            with self.assertRaises(ValueError):save_recordings([Segment(frames,times.tolist(),'manual')],'phrase',out,'manual')


class LiveBoundaryTests(unittest.TestCase):
    def test_manual_recognizer_classifies_once_after_finish(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock,patch
        import recognizer
        detector=MagicMock()
        detector.find_position.return_value=[('Left',[[i,100+i,200+i] for i in range(21)])]
        @contextmanager
        def fake_camera(index):yield MagicMock(),detector
        predictor=MagicMock();predictor.predict.return_value=('phrase',.9)
        frame=np.zeros((480,640,3),dtype=np.uint8)
        with patch.object(recognizer,'open_camera',fake_camera),patch.object(recognizer,'Predictor',return_value=predictor),patch.object(recognizer,'read_frame',return_value=frame),patch.object(recognizer.cv2,'imshow'),patch.object(recognizer.cv2,'getWindowProperty',return_value=1),patch.object(recognizer.cv2,'waitKey',side_effect=[32]+[-1]*89+[32,ord('q')]),patch.object(recognizer.time,'monotonic',side_effect=[i/30 for i in range(92)]):
            recognizer.run_recognizer('unused.joblib')
        predictor.predict.assert_called_once()
        self.assertEqual(predictor.predict.call_args.args[0].shape,(15,84))


class DesktopSettingsTests(unittest.TestCase):
    def test_settings_are_forwarded_to_both_camera_actions(self):
        from desktop import Desktop
        from unittest.mock import MagicMock,patch
        class Value:
            def __init__(self,v):self.v=v
            def get(self):return self.v
        app=Desktop.__new__(Desktop)
        for key,val in {'camera':'1','width':'','height':'','max_seconds':'8','idle_seconds':'1.1','motion_threshold':'1.6','mode':'manual','threshold':'.6','engine':'normalized','label':'phrase','samples':'30'}.items():
            setattr(app,key,Value(val))
        app.run=MagicMock()
        app.recognize()
        args=app.run.call_args.args[0]
        self.assertIn('--mode',args);self.assertEqual(args[args.index('--max-seconds')+1],'8.0')
        self.assertEqual(args[args.index('--idle-seconds')+1],'1.1')
        with patch('desktop.filedialog.asksaveasfilename',return_value='/tmp/new.csv'):
            app.collect()
        args=app.run.call_args.args[0]
        self.assertEqual(args[0],'collect');self.assertEqual(args[args.index('--motion-threshold')+1],'1.6')
        self.assertEqual(args[args.index('--max-seconds')+1],'8.0')


class StopRegressionTests(unittest.TestCase):
    def recording(self):
        c=SignCapture(max_seconds=8)
        c.update(hand(),0,toggle=True)
        for i in range(1,21):c.update(hand(100+i),i*.05)
        return c

    def test_finish_without_hands_uses_completed_frames(self):
        c=self.recording()
        s=c.update(None,1.05,toggle=True)
        self.assertIsNotNone(s);self.assertEqual(len(s.frames),21)
        self.assertAlmostEqual(s.duration,1);self.assertFalse(c.active)

    def test_manual_long_gap_preserves_buffer_for_stop(self):
        c=self.recording()
        c.update(None,1.1);c.update(None,1.5)
        self.assertTrue(c.active);self.assertTrue(c.interrupted)
        self.assertIn('Tracking paused',c.status)
        s=c.update(None,2,toggle=True)
        self.assertIsNotNone(s);self.assertEqual(len(s.frames),21)

    def test_returning_hands_do_not_stitch_manual_gap(self):
        c=self.recording()
        c.update(None,1.1);c.update(hand(500),1.5)
        self.assertTrue(c.interrupted)
        c.update(hand(600),1.6)
        s=c.update(hand(700),2,toggle=True)
        self.assertEqual(len(s.frames),21)
        self.assertEqual(s.frames[-1][0],120)

    def test_timeout_without_hands_finishes_buffer(self):
        c=self.recording()
        c.update(None,1.1);s=c.update(None,8.1)
        self.assertIsNotNone(s);self.assertEqual(s.reason,'maximum duration')

    def test_stop_too_short_still_leaves_recording_mode(self):
        c=SignCapture();c.update(hand(),0,toggle=True)
        self.assertIsNone(c.update(None,.1,toggle=True))
        self.assertFalse(c.active);self.assertIn('Too short',c.status)


class StopCameraRegression(unittest.TestCase):
    def test_space_and_enter_finish_after_hands_leave_camera(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock,patch
        import recognizer
        for stop_key in (32,13):
            with self.subTest(stop_key=stop_key):
                detector=MagicMock()
                detector.find_position.side_effect=[[('Left',[[i,100+i,200+i] for i in range(21)])]]*21+[[]]*21
                @contextmanager
                def fake_camera(index):yield MagicMock(),detector
                predictor=MagicMock();predictor.predict.return_value=('phrase',.9)
                frame=np.zeros((480,640,3),dtype=np.uint8)
                with patch.object(recognizer,'open_camera',fake_camera),patch.object(recognizer,'Predictor',return_value=predictor),patch.object(recognizer,'read_frame',return_value=frame),patch.object(recognizer.cv2,'imshow'),patch.object(recognizer.cv2,'getWindowProperty',return_value=1),patch.object(recognizer.cv2,'waitKey',side_effect=[32]+[-1]*39+[stop_key,ord('q')]),patch.object(recognizer.time,'monotonic',side_effect=[i*.05 for i in range(42)]):
                    recognizer.run_recognizer('unused.joblib')
                predictor.predict.assert_called_once()
                self.assertEqual(predictor.predict.call_args.args[0].shape,(15,84))

if __name__=='__main__':unittest.main()
