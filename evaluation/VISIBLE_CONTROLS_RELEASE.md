# Visible Start/Finish controls and armed capture

The supplied screenshot shows “Show your hands before starting,” which is the earlier non-recording state. The preview window remained open even when recording was not active. This update makes the state explicit rather than treating preview as evidence that a sample is being recorded.

Both camera windows now show Ready, Waiting for hands, Recording or Paused, plus Start sign, Finish sign, Discard and Close buttons. Start without visible hands arms capture; the first visible hand begins recording. Finish cancels an armed start or finishes an active/paused recording, using the buffered frames if no hands are visible. Explicit mouse Finish cannot restart an already finished recording. Finishing returns to Ready; Close exits the preview. Keyboard shortcuts are retained.

Training coordinates are extracted before the display-only control strip is appended, so the added controls do not change model coordinates or training data.

Validation: **47 discovered tests, 45 passed and 2 optional PyTorch checks skipped.** New tests cover armed start, hand arrival, mouse finish after tracking loss, disabled Finish, cancelling an armed start, discard/close, and keyboard alternatives. Previous pipeline, recording, database and normalization tests pass.

A real OpenCV window was rendered under Xvfb and visually inspected. Mouse events were injected through XTest/X11 (not by invoking the callback directly): Start -> Waiting for hands -> Recording -> Finish with no hands -> Ready -> Close passed. This validates the rendered mouse-event path on the Linux OpenCV backend. A physical webcam and the native macOS backend were not available for testing.

The model, CSV and dependencies are unchanged. Copy camera_controls.py (new), segmentation.py, collector.py, recognizer.py and desktop.py into an existing installation; preserve your own CSV, models and recordings. No retraining is required. Earlier release notes describe prior versions; README.md describes the current controls.
