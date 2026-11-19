"""Versioned legacy tensor loader; signer-aware split only when IDs exist."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, path, indices=None, classes=None):
        with np.load(path, allow_pickle=False) as data:
            x = data['x'].astype(np.float32)
            labels = data['labels'].astype(str)
        if x.ndim != 3 or x.shape[1:] != (15, 88) or not np.isfinite(x).all():
            raise ValueError('Expected finite (N,15,88) tensors')
        self.classes = [str(c) for c in (sorted(set(labels)) if classes is None else classes)]
        unknown = set(labels) - set(self.classes)
        if unknown: raise ValueError(f'Unknown labels: {unknown}')
        ids = np.arange(len(labels)) if indices is None else np.asarray(indices)
        self.x = torch.from_numpy(x[ids].copy())
        self.y = torch.tensor([self.classes.index(v) for v in labels[ids]], dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def split_indices(labels, seed=42, test_fraction=0.2, groups=None):
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    idx = np.arange(len(labels))
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(labels): raise ValueError('groups length mismatch')
        tr, val = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(idx, labels, groups))
        if set(groups[tr]) & set(groups[val]): raise AssertionError('Group leakage')
        return tr, val
    counts = np.unique(labels, return_counts=True)[1]
    stratify = labels if counts.min() >= 2 else None
    return train_test_split(idx, test_size=test_fraction, random_state=seed, stratify=stratify)
