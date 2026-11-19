import unittest
from unittest.mock import patch
import numpy as np
import cv2
from segmentation import SignCapture
from camera_controls import CameraControls


def hand():
    a=np.zeros((2,21,2));a[0]=[100,100];a[0,9]=[100,120]
    return a.reshape(84)

class VisibleControlTests(unittest.TestCase):
    def show(self,ui,capture):
        with patch('camera_controls.cv2.imshow'),patch('camera_controls.cv2.setMouseCallback'):
            ui.show(np.zeros((480,640,3),np.uint8),capture)
    def click(self,ui,action):
        x1,y1,x2,y2=ui.boxes[action]
        ui.on_mouse(cv2.EVENT_LBUTTONUP,(x1+x2)//2,(y1+y2)//2,0,None)
        return ui.command(-1)

    def test_start_arms_then_automatically_begins_when_hands_appear(self):
        c=SignCapture();ui=CameraControls('test');self.show(ui,c)
        cmd=self.click(ui,'start')
        c.update(None,0,toggle=ui.should_toggle(cmd,c))
        self.assertTrue(c.armed);self.assertFalse(c.active)
        c.update(None,.1);c.update(hand(),.2)
        self.assertFalse(c.armed);self.assertTrue(c.active)

    def test_mouse_finish_without_keyboard_or_visible_hands(self):
        c=SignCapture();ui=CameraControls('test')
        c.update(hand(),0,toggle=True)
        for i in range(1,21):c.update(hand(),i*.05)
        c.update(None,1.1);c.update(None,1.5)
        self.show(ui,c);cmd=self.click(ui,'finish')
        s=c.update(None,1.6,toggle=ui.should_toggle(cmd,c))
        self.assertIsNotNone(s);self.assertFalse(c.active)
        self.assertEqual(len(s.frames),21)

    def test_disabled_finish_cannot_start_a_recording(self):
        c=SignCapture();ui=CameraControls('test');self.show(ui,c)
        self.assertIsNone(self.click(ui,'finish'))
        self.assertFalse(ui.should_toggle('finish',c))

    def test_cancel_armed_start(self):
        c=SignCapture();c.update(None,0,toggle=True)
        ui=CameraControls('test');self.show(ui,c)
        cmd=self.click(ui,'finish');c.update(None,.1,toggle=ui.should_toggle(cmd,c))
        self.assertFalse(c.active);self.assertFalse(c.armed)

    def test_discard_and_close(self):
        c=SignCapture();c.update(hand(),0,toggle=True)
        ui=CameraControls('test');self.show(ui,c)
        self.assertEqual(self.click(ui,'discard'),'discard')
        self.assertEqual(self.click(ui,'quit'),'quit')

    def test_keyboard_alternatives_remain(self):
        ui=CameraControls('test')
        for key in [32,13,10]:self.assertEqual(ui.command(key),'toggle')
        self.assertEqual(ui.command(27),'discard');self.assertEqual(ui.command(ord('q')),'quit')
        self.assertIsNone(ui.command(-1))

if __name__=='__main__':unittest.main()
