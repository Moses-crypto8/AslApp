"""Camera lifecycle and optional training-coordinate mapping."""
from contextlib import contextmanager
import cv2
import hand_detector2 as hdm

@contextmanager
def open_camera(index):
    cap = cv2.VideoCapture(index)
    detector = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Camera {index} unavailable. Check camera permission and close other camera apps.')
        detector = hdm.handDetector()
        yield cap, detector
    finally:
        cap.release()
        if detector is not None:
            detector.hands.close()
        cv2.destroyAllWindows()

def read_frame(cap, reference_size=None):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Camera stopped returning frames. Check camera connection/permission.')
    frame = cv2.flip(frame, 1)
    # Legacy features are raw pixels. Only resize when the user knows the
    # original training resolution; do not guess it from landmark extrema.
    if reference_size:
        frame = cv2.resize(frame, reference_size)
    return frame
