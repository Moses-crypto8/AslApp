# ASL Studio — complete recordings and sign boundaries

The default classifier now recognizes **normalized handshape and motion**, rather than relying on absolute pixel placement. The CSV remains raw landmark coordinates; normalization is embedded in the saved model and applied identically during training and live inference.

## Visible camera controls (latest update)

The camera window now has **Start sign**, **Finish sign**, **Discard** and **Close** buttons. Keyboard shortcuts remain available. The state strip makes the difference explicit:

- **READY:** camera preview only; no sample is being recorded.
- **WAITING FOR HANDS:** Start was pressed without visible hands. Raise your hands and recording begins automatically. Finish cancels this pending start.
- **RECORDING:** frames are being captured. Click **Finish sign** to classify/save the completed sign.
- **PAUSED:** tracking was lost in manual mode; captured frames are kept. Finish uses those frames; Discard rejects them.

**Finishing a sign does not close the camera window.** It returns to Ready for the next sign, with the prediction shown. Use Close to exit recognition, or save completed collection samples and exit. A sample that is too short is rejected with an explanation.

To update your existing installation, replace these five code files from the package: `camera_controls.py` (new), `segmentation.py`, `collector.py`, `recognizer.py`, and `desktop.py`. Keep your local `sign_language.csv`, model files and recordings. No retraining is required.

## SPACE/Enter stop fix

Stop now works even when no hand is detected. In manual mode, a long tracking interruption freezes and preserves the existing recording until you finish or discard it. The second SPACE (or Enter) finishes the saved frames; a short/empty sample exits recording mode and displays a rejection reason. No retraining is needed for this fix.

Keyboard shortcuts require camera-window focus; clickable camera controls are also available. Keep existing `sign_language.csv`, models and recordings when updating: replace only `camera_controls.py`, `segmentation.py`, `collector.py`, `recognizer.py` and `desktop.py` from this package if you have already collected/trained locally.

## Complete-sign recording and recognition

Both collection and recognition now use full recording boundaries instead of guessing every arbitrary 15 camera frames:

- **Manual mode (default):** show your hands, press SPACE, perform the full sign/phrase, then press SPACE or Enter again. Recognition runs once on the completed segment. Esc discards the current segment; q exits and saves only completed collection samples.
- **Duration:** default maximum 6 seconds, adjustable up to 30 seconds. Minimum accepted recording is 0.35 seconds and three frames. Maximum duration finishes the current valid recording automatically, so set it longer than your longest phrase. An unfinished recording is discarded on q.
- **Auto mode (optional):** hand arrival or motion starts capture; about 0.8 seconds of low motion finishes it. A short pre-roll retains onset frames, and trailing pause is shortened. Static held signs can finish; a cooldown avoids repeated classifications of an unchanged hand. This is a tunable heuristic, not learned linguistic segmentation. It can split a phrase containing a long pause or include setup movements. Manual mode is recommended for phrases and collection.
- **Tracking:** manual mode preserves the captured frames if you move your hands away to press stop. A gap of 0.25 seconds freezes capture and shows “Tracking paused”; SPACE/Enter finishes the saved portion, while Esc discards it. Returning hands are not joined across that gap. Auto mode still discards long tracking gaps. Short interruptions are tolerated.
- **Shared model input:** each completed recording is sampled across its full elapsed duration into the existing model's 15 input frames. Collection and recognition use exactly the same function. This preserves compatibility with your CSV and pretrained models. Fifteen input frames can still miss fine details in long phrases; this is not a continuous ASL translator.
- **Raw data retained:** collection also writes `NAME.recordings.npz` (all raw frames, timestamps, sample offsets and labels) and `NAME.recordings.json` (duration, frame count, end reason, session ID) alongside `NAME.csv`. Keep all three files. Import the CSV into the current training workflow; companion files retain data for future variable-length models. Current training does not use duration or session IDs as features/groups.

GUI settings: Boundaries, Max seconds, Auto pause and Motion sensitivity. Collection and recognition share these settings. Start in manual mode to judge recognition on complete signs before tuning automatic mode.

```bash
python app.py recognize --mode manual --max-seconds 8
python app.py collect --label "hi, how are you?" --out updates/phrase.csv --samples 30 --mode manual --max-seconds 8
python app.py recognize --mode auto --idle-seconds 1.0 --motion-threshold 1.2 --max-seconds 8
```

Lower motion threshold makes automatic capture more sensitive; longer pause tolerance is less likely to split a phrase but adds delay. Old recordings remain compatible, but their missing duration/boundary information cannot be reconstructed. Collect complete examples, import their CSV, and retrain before judging performance on newly recorded long phrases.

**No new live-accuracy improvement is claimed:** model weights and legacy accuracy are unchanged. Boundary logic is tested with synthetic and mocked recordings; real-camera performance must be checked locally.

## Position and size normalization

For each 15-frame sequence:

1. Detect which hands are present, preserving zero-filled missing hands.
2. Choose the most consistently visible hand and its first visible wrist as a fixed reference position.
3. Measure the median palm length across visible hands/frames as a single sequence-wide size reference.
4. Express all landmarks relative to that position and size. This removes a uniform change of location or scale across the entire sign.
5. Also provide wrist-centered, palm-scaled finger landmarks for handshape, plus hand-presence masks.

**The reference position stays fixed throughout the sequence.** This preserves wrist trajectories and relative positions between the hands. It does not recenter every frame and erase motion.

Default estimator: 300-tree Extra Trees classifier, chosen using three-fold validation within the original training partition. No PyTorch is needed for the default model. The internal `rf` engine name is retained as an alias for compatibility; `normalized` is the current name.

## Measured results

| Model | Legacy diagnostic score | Correct / 118 |
|---|---:|---:|
| New default: normalized handshape + motion | **95.76%** | **113** |
| Original pixel-placement baseline | 97.46% | 115 |
| Older optional GRU | 96.61% | 114 |

The normalized model retained **100% prediction consistency** across all 118 holdout samples under each of five synthetic transformations: two translations, half-size plus translation, 1.6× size plus translation, and double size. Accuracy remained 95.76% in every case. This means the predictions stayed the same; it does not mean every sign was classified correctly.

These transformations operate on recorded landmarks, not camera images. They do not test MediaPipe tracking when hands are small, blurred, occluded, clipped, or at a different perspective. They can include coordinates outside an actual image and are mathematical robustness tests, not physical frame-coverage tests.

The dataset has 589 samples and 17 classes; training uses 471 samples and evaluation 118, seed 42, stratified 80/20. There are no exact duplicated feature rows across the split. This legacy holdout has been used during development; it is a diagnostic benchmark, not a new untouched test set. Classifier selection was confined to the 471 training samples. New-signer and real-camera accuracy remain unmeasured. Detailed reports are in `evaluation/` and `models/asl_model.json`.

## Start

Use **Python 3.11 or 3.12**. Extract the full ZIP, open Terminal in `asl_updatable_app`, then run:

```bash
python3 bootstrap.py
```

First launch creates `.venv` and installs requirements; internet is needed for setup. Later launches reuse it. On macOS, `start.command` finds Python 3.12/3.11 when available. Allow Terminal/Python camera access in System Settings → Privacy & Security → Camera. Python must include Tk (included with python.org macOS/Windows installers).

Windows: install Python 3.12, then double-click `start.bat` or run `py -3.12 bootstrap.py`.

This is a Python source app with pretrained models, not a signed native installer.

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py gui
```

Windows activation: `.venv\Scripts\activate`. Install the single listed OpenCV distribution; do not install another OpenCV or headless build into this environment.

## Use

- Leave the engine set to **normalized** and click **Start recognition**. Press **q** in the camera window to close it. Try camera 1 if camera 0 is unavailable.
- Position and size normalization are automatic. The threshold controls confidence acceptance; the displayed confidence is not measured accuracy.
- Leave Training width/height blank for native camera pixels. Uniform resizing does not require matching training dimensions with the normalized model. Avoid nonuniform stretching or changing aspect ratio. The older pixel baseline remains sensitive to resolution and framing.
- Enter a new sign label and sample count, then **Collect samples**. Choose a new CSV filename. SPACE starts a complete recording and SPACE again finishes it; q saves completed recordings and exits. Long tracking gaps discard the current recording.
- **Import CSV + retrain** backs up and updates the database, then trains the normalized model. **Train model** retrains the current database. Capture enough samples for every term.
- You no longer need duplicate recordings solely to represent an ideal uniform translation or scale change. You still need examples of differences in execution, signer anatomy, speed, viewpoint, lighting, tracking quality, and missing hands.

## Limits

Uniform translation and positive uniform scaling are normalized. Rotation, mirror changes, perspective changes, clipped hands, and arbitrary independent movement/resizing of each hand are not made equivalent. Movement during a sign is intentionally retained. A change in apparent size *during* a recording is not removed globally.

There are no face/body landmarks. ASL signs can differ by body-relative location; discarding absolute camera location cannot recover forehead-versus-chin context. This feature does not provide body-relative understanding or continuous ASL sentence translation. It recognizes only the 17 existing isolated labels and newly trained labels; unsupported signs may still receive confident guesses.

## Commands

```bash
python app.py recognize
python app.py recognize --engine normalized --camera 0 --threshold 0.55
python app.py collect --label "good morning" --out updates/good_morning.csv --samples 30
python app.py update-db --update updates/good_morning.csv --retrain
python app.py train
python app.py terms
python app.py evaluate
python evaluate_invariance.py
python -m unittest discover -s tests -v
```

Default paths resolve next to `app.py`; explicitly supplied relative paths resolve from your current directory. Only load trusted model files. The raw CSV is unchanged. The original pixel model remains available:

```bash
python app.py recognize --model models/asl_pixel_baseline.joblib
```

That explicit baseline bypasses normalization because its saved model uses the original raw features.

Optional older GRU (per-frame centering loses absolute wrist trajectories; it is separate from the new normalized model):

```bash
python -m pip install -r requirements-temporal.txt
python app.py recognize --engine gru
python app.py train-temporal --epochs 12
```

Use Python with Tk and a graphical desktop. For MediaPipe `solutions` errors, activate a clean environment and install `requirements.txt`. If setup was interrupted, rerun bootstrap. On Linux, Tk and OpenGL runtime packages may need installation through your distribution. Local webcam and native macOS/Windows launcher testing are still required.
