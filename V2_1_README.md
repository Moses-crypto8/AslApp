# V2.1 — Isolated-sign GRU training baseline

This is a **research baseline**, not continuous ASL recognition or translation.
The original V1 application and model remain unchanged.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m v2.train_v2 --epochs 12
```

Outputs: `v2_runs/gru_baseline/best_gru.pt`, `metrics.json`, `labels.json`.
Checkpoint is PyTorch only; no Core ML export or iOS integration yet.

## Evaluation caveat

The legacy CSV lacks signer/session identifiers. The included stratified random
split is **diagnostic only** and may leak signer/session information across sets.
Do not interpret its score as performance on new people. When new data includes
signer IDs, provide a `.npy` array aligned with the NPZ rows using `--groups`.
A production benchmark should reserve entirely unseen signers and recording sessions.

## Data limitations and next work

* Current input is 15 frames of 88 hand-only features; facial expression and body
  pose are **not** represented.
* The legacy transform wrist-centers each frame, removing absolute wrist trajectory.
  Before collecting V2 recordings, add a motion-preserving feature representation
  and timestamped raw landmarks, along with consent-aware signer/session metadata.
* GRU is an isolated-sign classifier; continuous recognition requires segmentation
  or sequence decoding and a separate sentence-level translation dataset.
* Cloud translation should be opt-in and never receive camera frames by default.
