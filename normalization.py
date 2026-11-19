"""Sequence-level translation/scale normalization, preserving motion paths."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

REPRESENTATION = 'sequence_origin_palm_scale_plus_handshape_v1'

def normalize_sequence(sequence):
    a = np.asarray(sequence, dtype=np.float64)
    if a.shape != (15,84) or not np.isfinite(a).all():
        raise ValueError('Expected finite (15,84) landmarks.')
    hands = a.reshape(15,2,21,2)
    present = np.any(hands != 0, axis=(2,3))
    if not present.any():
        return np.zeros((15,170), dtype=np.float64)
    # Use a single anchor and scale for the ENTIRE sequence, never recenter
    # each frame: movement relative to the starting position must survive.
    reference = int(present.sum(axis=0).argmax())
    first = int(np.flatnonzero(present[:,reference])[0])
    origin = hands[first,reference,0].copy()
    palm_lengths = np.linalg.norm(hands[:,:,9] - hands[:,:,0], axis=-1)
    valid = palm_lengths[present & (palm_lengths > 1e-8)]
    if not len(valid):
        raise ValueError('Cannot normalize a sequence with no measurable palm size.')
    scale = float(np.median(valid))
    normalized = (hands - origin) / scale
    normalized[~present] = 0
    # Also expose local handshape so distance between the hands and wrist
    # travel do not dominate the classifier's representation of finger shape.
    safe_lengths = np.where(palm_lengths > 1e-8, palm_lengths, scale)
    local = (hands - hands[:,:,0:1,:]) / safe_lengths[:,:,None,None]
    local[~present] = 0
    return np.concatenate([normalized.reshape(15,84),local.reshape(15,84),present.astype(float)], axis=1)

class SequenceNormalizer(TransformerMixin, BaseEstimator):
    """Identical preprocessing embedded in saved training/inference pipeline."""
    def fit(self, X, y=None):
        self.n_features_in_ = 1260
        return self
    def transform(self, X):
        a = np.asarray(X, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 1260:
            raise ValueError('Expected rows of 1260 raw landmark coordinates.')
        return np.stack([normalize_sequence(s).reshape(-1) for s in a.reshape(-1,15,84)])

def relocate_sequence(sequence, scale=1.0, offset=(0,0)):
    """Synthetic diagnostic only: transform visible hands; keep absent hands zero."""
    hands = np.asarray(sequence,dtype=np.float64).reshape(15,2,21,2).copy()
    present = np.any(hands != 0,axis=(2,3))
    hands[present] = hands[present]*scale + np.asarray(offset)
    return hands.reshape(15,84)
