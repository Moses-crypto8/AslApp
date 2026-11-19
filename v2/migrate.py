"""Convert V1 flat CSV into V2 feature tensors without overwriting V1."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessing import FRAMES, INPUT_DIM, OUTPUT_DIM, transform

def migrate(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError('Missing label column')
    # Respect V1's original column order; verify shape and naming.
    cols = [c for c in df if c != 'label' and not c.startswith('Unnamed:')]
    if len(cols) != FRAMES * INPUT_DIM:
        raise ValueError(f'Expected {FRAMES * INPUT_DIM} features; got {len(cols)}')
    expected = [f'{frame}, {hand}, {point}, {axis}' for frame in range(FRAMES)
                for hand in ('L', 'R') for point in range(21) for axis in ('x', 'y')]
    if set(cols) != set(expected):
        raise ValueError('Unexpected feature columns in V1 CSV')
    raw = df.loc[:, expected].to_numpy(dtype=np.float32).reshape(-1, FRAMES, INPUT_DIM)
    x = np.stack([transform(s) for s in raw])
    labels = df['label'].astype(str).to_numpy(dtype='U128')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'legacy_features_v2.npz', x=x, labels=labels)
    metadata = {'samples': len(df), 'classes': sorted(set(labels)), 'frames': FRAMES,
                'features_per_frame': OUTPUT_DIM, 'source': str(csv_path),
                'source_type': 'legacy_v1', 'signer_ids_available': False,
                'evaluation_warning': 'Signer-independent validation impossible without signer/session metadata. Do not interpret random-split accuracy as generalization.',
                'limitations': ['Original frame-wise absolute wrist motion is lost after centering.', 'Legacy data has no body or face landmarks.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='sign_language.csv')
    parser.add_argument('--out', default='v2_data')
    args = parser.parse_args()
    print(json.dumps(migrate(args.csv, args.out), indent=2))
