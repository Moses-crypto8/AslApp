from __future__ import annotations
from pathlib import Path
import json
import time
import uuid
import numpy as np
import cv2
from camera import open_camera, read_frame
from features import landmarks_to_frame_vector, sequences_to_dataframe
from segmentation import SignCapture,resample_segment
from camera_controls import CameraControls


def save_recordings(segments,label,output_csv,mode):
    """Save raw frames/timestamps and a compatible complete-sign 15-frame CSV."""
    output_csv=Path(output_csv)
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    npz=output_csv.with_suffix('.recordings.npz');meta=output_csv.with_suffix('.recordings.json')
    if any(p.exists() for p in (output_csv,npz,meta)):raise ValueError('Recording output exists; choose a new filename.')
    sequences=[resample_segment(s.frames,s.times).tolist() for s in segments]
    offsets=np.r_[0,np.cumsum([len(s.frames) for s in segments])]
    session_id=str(uuid.uuid4())
    np.savez_compressed(npz,frames=np.concatenate([s.frames for s in segments]).astype(np.float64),
                        times=np.concatenate([np.asarray(s.times)-s.times[0] for s in segments]),
                        offsets=offsets,labels=np.asarray([label]*len(segments)),session_id=np.asarray(session_id))
    meta.write_text(json.dumps({'format_version':1,'session_id':session_id,'label':label,'mode':mode,
        'csv_resampled_frames':15,'raw_recordings':npz.name,
        'samples':[{'duration_seconds':s.duration,'frames':len(s.frames),'end_reason':s.reason} for s in segments]},indent=2))
    sequences_to_dataframe(sequences,label).to_csv(output_csv,index=False)
    return output_csv


def collect_samples(label,output_csv,camera_index=0,target_samples=25,reference_size=None,
                    max_seconds=6.,min_seconds=.35,mode='manual',idle_seconds=.8,motion_threshold=1.2):
    label=label.strip();output_csv=Path(output_csv)
    if not label or target_samples<1:raise ValueError('Provide a nonempty label and positive sample count.')
    if any(p.exists() for p in (output_csv,output_csv.with_suffix('.recordings.npz'),output_csv.with_suffix('.recordings.json'))):
        raise ValueError('Output already exists. Choose a new filename.')
    capture=SignCapture(mode,max_seconds,min_seconds,idle_seconds,motion_threshold=motion_threshold)
    controls=CameraControls('ASL Sample Collector')
    segments=[];error=None
    try:
        with open_camera(camera_index) as (cap,detector):
            while len(segments)<target_samples:
                img=read_frame(cap,reference_size)
                detector.find_hands(img,draw=True);hands=detector.find_position(img,draw=False)
                vector=landmarks_to_frame_vector(hands) if hands else None
                cv2.putText(img,f'{label}: {len(segments)}/{target_samples} | {mode}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.putText(img,capture.status,(20,70),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                cv2.putText(img,'SPACE/Enter start/finish | Esc discard | q save completed samples',(20,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                controls.show(img,capture)
                key=cv2.waitKey(1)&0xff
                command=controls.command(key)
                if command=='quit' or cv2.getWindowProperty('ASL Sample Collector',cv2.WND_PROP_VISIBLE)<1:break
                segment=capture.update(vector,time.monotonic(),toggle=controls.should_toggle(command,capture),cancel=command=='discard')
                if segment:
                    segments.append(segment);print(f'Captured {segment.duration:.2f}s / {len(segment.frames)} frames',flush=True)
    except (RuntimeError,KeyboardInterrupt) as exc:error=exc
    if segments:
        save_recordings(segments,label,output_csv,mode)
        print(f'Saved {len(segments)} complete recordings to {output_csv} (raw frames retained alongside CSV)',flush=True)
    else:print('No completed recordings saved. SPACE starts and finishes a recording.',flush=True)
    if error:raise error
    return output_csv if segments else None
