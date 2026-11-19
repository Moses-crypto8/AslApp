"""V2 hand-sequence features. V1 data is pixel-based, not metric coordinates."""
import numpy as np

FRAMES = 15
POINTS = 21
INPUT_DIM = 84
OUTPUT_DIM = 88  # 84 coordinates + 2 masks + 2 wrist-relative displacement coords

def transform(sequence):
    """(15,84) -> (15,88). Per-hand wrist centering and palm scale.

    Missing hands remain zeros; mask differentiates absence from centered wrists.
    Keep inter-hand wrist offset (normalized by reference hand scale) to retain
    relative positioning. Absolute trajectory is NOT preserved by this representation.
    """
    a = np.asarray(sequence, dtype=np.float32)
    if a.shape != (FRAMES, INPUT_DIM) or not np.isfinite(a).all():
        raise ValueError('Expected finite (15,84) sequence')
    hands = a.reshape(FRAMES, 2, POINTS, 2)
    present = np.any(hands != 0, axis=(2,3)).astype(np.float32)
    wrists = hands[:, :, 0, :].copy()
    # Palm scale: wrist to middle-finger MCP (landmark 9).
    scale = np.linalg.norm(hands[:, :, 9, :] - wrists, axis=-1)
    scale = np.maximum(scale, 1.0)
    centered = (hands - wrists[:, :, None, :]) / scale[:, :, None, None]
    centered *= present[:, :, None, None]
    relative = (wrists[:, 1] - wrists[:, 0]) / scale[:, 0, None]
    relative *= (present[:, 0] * present[:, 1])[:, None]
    return np.concatenate([centered.reshape(FRAMES, INPUT_DIM), present, relative], axis=1).astype(np.float32)
