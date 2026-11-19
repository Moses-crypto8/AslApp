from __future__ import annotations
import time
import cv2
from camera import open_camera, read_frame
from features import landmarks_to_frame_vector
from inference import Predictor
from segmentation import SignCapture,resample_segment
from camera_controls import CameraControls

def run_recognizer(model_path='models/asl_model.joblib',camera_index=0,
                   confidence_threshold=.55,engine='normalized',reference_size=None,
                   max_seconds=6.,min_seconds=.35,mode='manual',idle_seconds=.8,motion_threshold=1.2):
    if not 0<=confidence_threshold<=1:raise ValueError('Confidence threshold must be between 0 and 1.')
    predictor=Predictor(model_path,engine)
    capture=SignCapture(mode,max_seconds,min_seconds,idle_seconds,motion_threshold=motion_threshold)
    controls=CameraControls('ASL Recognizer')
    last='Click Start sign or press SPACE';last_time=None
    with open_camera(camera_index) as (cap,detector):
        while True:
            img=read_frame(cap,reference_size)
            detector.find_hands(img,draw=True);hands=detector.find_position(img,draw=False)
            vector=landmarks_to_frame_vector(hands) if hands else None
            now=time.monotonic()
            if last_time is not None and now-last_time>4:last='SPACE: start next sign'
            cv2.putText(img,'Recording - click Finish sign when done' if capture.active else ('Waiting for hands' if capture.armed else last),(20,40),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
            cv2.putText(img,capture.status,(20,75),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            cv2.putText(img,f'{mode} | SPACE/Enter start/finish | Esc discard | q quit',(20,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            controls.show(img,capture)
            key=cv2.waitKey(1)&0xff
            command=controls.command(key)
            if command=='quit' or cv2.getWindowProperty('ASL Recognizer',cv2.WND_PROP_VISIBLE)<1:break
            segment=capture.update(vector,now,toggle=controls.should_toggle(command,capture),cancel=command=='discard')
            if segment:
                label,confidence=predictor.predict(resample_segment(segment.frames,segment.times))
                last=f'{label} ({confidence:.0%})' if label and confidence>=confidence_threshold else f'Unsure ({confidence:.0%})'
                last_time=now;print(f'{last} | {segment.duration:.2f}s / {len(segment.frames)} frames',flush=True)
