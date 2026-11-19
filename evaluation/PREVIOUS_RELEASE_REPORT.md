# ASL Studio verification report

## Outcome

The default Random Forest matches the supplied prototype's recorded **97.4576%** score: **115 correct out of 118 held-out samples** across 17 terms. The included optional GRU checkpoint reproduces **96.6102%**, 114/118. No hyperparameter search was performed to increase the default score.

## Evaluation procedure

- Dataset: supplied `sign_language.csv`, unchanged; 589 samples, 1260 features and a label after dropping the accidental index column.
- Stratified 80/20 split, random state 42; 471 training / 118 holdout samples.
- RF: 200 trees, balanced class weights, random state 42. Shipped RF uses the training partition only.
- No exact duplicate feature rows and no identical feature vectors spanning train and holdout.
- Score evidence: original metadata in `original_prototype.json`; reproduced metrics in `reproduced.json`; current default model metadata in `../models/asl_model.json`.
- Reports include per-term precision/recall, confusion matrices, indices, dataset SHA-256, and runtime versions.
- GRU checkpoint is evaluated against its supplied validation split and uses validation-selected weights. No independent GRU test set was supplied.

## Implemented and checked

- Desktop interface for recognition, RF/GRU training, term listing, collecting samples, and CSV import/retraining.
- Graphical smoke test under Xvfb: interface rendered and visually inspected, 97.46% score displayed, terms action completed, process log populated, controls returned to ready state.
- Default paths resolve relative to the application, so CLI launching from another directory works.
- Dependency line repaired; base and optional temporal requirements separated. Python 3.11–3.12 supported by the selected dependency versions; validation used Python 3.12.
- Base dependency environment: `pip check` reported no broken requirements; holdout retraining reproduced 97.46%. Base tests: 15 discovered, 13 passed and 2 optional-PyTorch skips.
- Full environment: **17 tests passed** with PyTorch available. Logs are retained in this directory.
- Real MediaPipe hand detector initialized and processed a blank image successfully.
- Mock-camera tests verified recognizer execution and a complete 15-frame collection/CSV save. Camera-unavailable test verified cleanup.
- Sequence gaps clear the current recording; no stitching across hand-tracking gaps.
- Canonical feature ordering checked, all-zero sequences rejected as no hands, malformed dimensions rejected.
- CSV updates checked for required schema, numeric finite coordinates, nonempty labels; invalid data leaves the database unchanged. Backups and atomic CSV/model file replacement added.
- GRU migration and one-epoch training smoke test completed in a temporary directory without changing the released checkpoint.
- macOS/Windows launcher files and setup instructions included. Launchers were inspected; actual macOS/Windows OS execution was not available here.

## What still needs local measurement

No physical webcam or human signing was available. Live signing accuracy, camera permission dialogs, and new-signer/new-session performance are **not verified**. GUI rendering and synthetic/mocked pipeline tests do not substitute for those tests.

The dataset has no original camera dimensions, signer IDs, or session IDs. Raw-pixel RF features depend on camera resolution and framing; the app provides optional training width/height controls without inventing those dimensions. Both collection and recognition mirror the camera, matching the supplied code. The model recognizes only its known isolated classes; it is not a general ASL sentence translator, and confidence is not a calibrated correctness guarantee.

To establish deployment accuracy, record a separate labeled test set across new sessions and signers, retain signer/session IDs, and evaluate the frozen models once on that set. The supplied data cannot establish that result.
