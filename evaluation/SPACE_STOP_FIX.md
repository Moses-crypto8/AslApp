# SPACE/Enter stop correction

Previously, stop depended on a hand being detected on the stop frame. Moving a hand toward the keyboard could cause the current sample to be discarded before SPACE was handled. Manual recordings also cleared automatically after a long tracking gap.

The stop command now takes priority over tracking availability and finishes the already captured frames. Manual mode freezes the buffer after a long tracking gap and displays an explicit pause message; returning hands are not joined across that gap. SPACE or Enter finishes that saved portion, while Esc discards it. Short/empty recordings still leave recording mode and show a rejection reason. The maximum duration also finishes a valid paused manual buffer. Automatic mode retains its tracking-loss discard behavior.

Both recognition and collection accept SPACE, Enter (CR), and Enter (LF). The camera window must have keyboard focus. No webcam hardware was available to reproduce the user's exact interaction, but controller and mocked-camera regressions cover the identified failure path.

Validation: 41 tests discovered, 39 passed, 2 optional PyTorch skips. Tests cover stopping without hands, stopping after a long tracking gap, avoiding stitching when hands return, timeout while paused, short-sample rejection, and both SPACE/Enter in a mocked recognition loop. Collection is tested with Enter and hands absent on the stop frame. Model weights, datasets, and dependencies are unchanged; retraining is not required.

To update an existing installation without overwriting newly collected data or trained models, copy only segmentation.py, collector.py, recognizer.py and desktop.py from the updated package into the app folder. Keep your own sign_language.csv, models, and recording files. Historical release reports describe earlier behavior; README.md and this report describe this correction.
