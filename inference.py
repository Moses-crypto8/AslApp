"""Shared, camera-independent inference used by live recognition and tests."""
from pathlib import Path
import numpy as np
import pandas as pd
from trainer import load_model

class Predictor:
    def __init__(self, path, engine='normalized'):
        self.engine = engine
        if engine in ('rf', 'normalized'):
            self.payload = load_model(path)
            self.classes = self.payload['classes']
        elif engine == 'gru':
            import torch
            from v2.gru_model import SignGRU
            self.torch = torch
            torch.set_num_threads(min(torch.get_num_threads(), 4))
            ck = torch.load(Path(path), map_location='cpu', weights_only=True)
            self.classes = ck['classes']
            self.model = SignGRU(input_size=ck['input_size'], hidden_size=ck['hidden_size'],
                                 layers=ck['layers'], num_classes=len(self.classes))
            self.model.load_state_dict(ck['state_dict'])
            self.model.eval()
        else:
            raise ValueError('Engine must be normalized (rf alias) or gru.')

    def predict(self, sequence):
        a = np.asarray(sequence, dtype=np.float64)
        if a.shape != (15,84) or not np.isfinite(a).all():
            raise ValueError('Expected finite (15,84) landmarks.')
        if not np.any(a):
            return None, 0.0
        if self.engine in ('rf', 'normalized'):
            from features import feature_columns
            row = pd.DataFrame(a.reshape(1,-1), columns=feature_columns())
            probabilities = self.payload['model'].predict_proba(row[self.payload['feature_columns']])[0]
        else:
            from v2.preprocessing import transform
            with self.torch.inference_mode():
                x = self.torch.from_numpy(transform(a)[None])
                probabilities = self.model(x).softmax(-1)[0].numpy()
        i = int(probabilities.argmax())
        return str(self.classes[i]), float(probabilities[i])

class SequenceBuffer:
    """Never join samples across a hand-tracking gap."""
    def __init__(self):
        self.frames = []
    def push(self, vector):
        if vector is None or not np.any(vector):
            self.frames.clear()
            return None
        self.frames.append(vector)
        if len(self.frames) == 15:
            result, self.frames = self.frames, []
            return result
        return None
