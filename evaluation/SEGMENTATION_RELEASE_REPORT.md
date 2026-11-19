# Recording duration and segmentation update

## Behavior

Manual complete-sign capture is now the default for both collection and recognition. SPACE starts and finishes a recording; Esc discards; q closes, saving completed collection samples only. Default maximum is 6 seconds (configurable up to 30), minimum 0.35 seconds and at least three observations. A maximum-duration boundary automatically finishes valid recordings.

Optional automatic boundaries use palm-normalized landmark speed, hand arrival, a 0.2-second pre-roll, and 0.8-second pause tolerance. Short trailing idle is removed, a cooldown limits repeated static predictions, and a missing-hand interval of 0.25 seconds discards the segment. Manual completion is still available in auto mode. This is a heuristic for separated signs, not a trained ASL boundary detector.

Shared elapsed-time resampling maps the whole segment to 15 model frames. Missing individual hands are represented by the nearest observation rather than linearly fabricating a partially present hand. Every raw capture and relative timestamp is retained in a companion NPZ; JSON contains sample durations, stop reasons, frame counts and session ID. CSV exports remain compatible with the existing model/database.

## Validation

35 tests discovered: **33 passed, 2 skipped** because optional PyTorch is not installed in this environment. Previous GRU code and weights were not changed. See `segmentation_test_results.txt` for exact results.

Tests include manual three-second/91-frame capture, static signs, automatic motion/pause capture, static cooldown, minimum/maximum duration, cancellation, tracking-loss rejection and recovery, short-gap tolerance, nonuniform timestamp resampling, missing-hand masks, raw-recording retention, and a mocked recognition loop verifying that a complete three-second sign triggers exactly one prediction. Existing collection, model-score, position/scale invariance and database tests pass.

New GUI controls pass capture-argument forwarding checks. The updated desktop was also rendered under Xvfb and visually inspected; the term-list action completed and the new manual/6-second settings were verified. All Python sources compile. No physical webcam or live signer is available here. The shipped model and dataset are unchanged: 95.76% remains a legacy recorded-data diagnostic, not a new live-accuracy estimate.

## Limits and next use

Start with manual mode. Record complete examples using the new collector, import the CSV, retrain, and assess fresh recordings. Existing 15-frame samples cannot recover lost timing. A 15-frame model input can undersample long phrases even when capture boundaries are correct. Raw recordings are retained for a future variable-length temporal model.

Automatic pauses can cut a held phrase or include hand setup motions; its thresholds need local tuning. Long tracking losses discard the active example; q discards incomplete examples. Face/body landmarks and linguistic sentence segmentation are outside this change. No improvement in measured live accuracy is claimed.
