# ASL V2 — milestone 1

The existing V1 app, model and CSV are preserved. This adds a **separate**
V2 preprocessing and legacy migration pipeline; no GRU is trained yet.

## Run

```bash
python -m pip install numpy pandas
python -m unittest discover -s tests -v
python -m v2.migrate --csv sign_language.csv --out v2_data
```

Outputs: `v2_data/legacy_features_v2.npz` (x shape N×15×88 and labels),
and `v2_data/metadata.json`.

## Important findings

* The legacy CSV contains an extra `Unnamed: 0` index column; migration ignores it. Check whether V1 training used this column as a feature.
* V1 uses 15 frames × 84 x/y **pixel coordinates** (not normalized positions).
* Legacy CSV does not record signer IDs, session IDs, FPS or raw video.
* The existing 97.46% random-split score cannot establish signer generalization.
* V2 centers each hand on its wrist and normalizes by wrist-to-middle-MCP
  distance; it keeps a normalized inter-hand offset and presence masks.
* This is an experimental representation: centering loses absolute wrist
  trajectories, which may be critical to some signs. Capture wrist motion
  separately in the next data collection version and compare representations.
* The old detector only appends frames **when hands are detected**, so its
  15-frame windows may represent inconsistent durations. V2 collection should
  capture fixed-rate frames including missing-hand masks.
* Model evaluation must use new recordings with signer/session metadata.

## Next milestone

Create a fixed-rate, metadata-rich collector, gather recordings from multiple
signers, then train and evaluate GRU vs Random Forest using grouped splits.
