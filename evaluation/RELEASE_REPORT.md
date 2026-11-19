# Position/scale normalization release

## Result

New default: **95.7627% (113/118)** on the legacy development holdout, compared with original pixel RF **97.4576% (115/118)**. The raw CSV is unchanged. Original RF and optional GRU models are retained as alternatives.

The default is a saved pipeline: sequence-level position/scale normalization plus local handshape and presence masks, followed by a 300-tree Extra Trees classifier. The representation is `sequence_origin_palm_scale_plus_handshape_v1`. A single fixed origin and scale per sequence preserve motion paths and inter-hand geometry while removing uniform translation and positive uniform scale.

## Selection and evaluation

The same seed-42 stratified split has 471 training and 118 evaluation samples. An initial sequence-normalized RF scored 90.68%; local handshape features were then added. Estimator selection for the enhanced representation used three-fold cross-validation strictly within the 471 training rows: Extra Trees 98.09%, RF 97.03%, SVM 93.84%. Extra Trees was selected by that inner validation score, then fitted to the 471 training rows. See `model_selection.json`.

Because the legacy holdout has been observed during development, **95.76% is a development diagnostic**, not an independent final test estimate. No unseen-signer, new-session, or physical camera accuracy is claimed. The former 97.46% accuracy does not describe the new default.

`position_scale_report.json` compares the normalized model and original pixel baseline on unchanged data and five synthetic transformations of all 118 evaluation sequences. The normalized model made identical predictions for every sample under every tested transform (590 transformed sequences); it remained 113/118 correct in every scenario. Mathematical feature invariance was also checked with randomized scales and translations. These are coordinate transformations, not actual camera images; they do not simulate tracking failures or field-of-view limits.

## Checks

- Full suite: 22 tests, including original tests and five normalization tests.
- Random translation/scale feature invariance, held-out prediction consistency, preserved motion, preserved inter-hand spacing, absent-hand masks, and degenerate-input rejection.
- Original baseline score regression, feature ordering, CSV validation/updates, camera failure cleanup, mocked recognition and collection, actual MediaPipe blank-frame processing, GRU checkpoint score reproduction.
- Desktop graphical smoke test and visual inspection performed for this update: correct new score and normalized engine displayed; term listing completed with controls reenabled.
- Classifier training and recognition share the same embedded preprocessing. CSV imports and retraining use the normalized pipeline automatically.

The raw coordinate file, vocabulary, and prior recordings are retained. No artificial shifted/scaled rows were mixed into train and test; invariance comes from preprocessing. Normalization is not rotation, mirror, perspective, face, or body-pose invariance. Body-relative sign location cannot be inferred from this dataset.

Reports from the previous release (97.46% default RF) are historical evidence, not the new default score; see `PREVIOUS_RELEASE_REPORT.md`. Current default metrics are `../models/asl_model.json`, and current tests are `test_results.txt`.
