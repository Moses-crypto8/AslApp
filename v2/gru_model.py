"""Compact temporal classifier for isolated signs (not ASL sentence translation)."""
import torch
from torch import nn

class SignGRU(nn.Module):
    def __init__(self, input_size=88, hidden_size=96, layers=2, num_classes=17, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(dropout),
                                  nn.Linear(hidden_size * 2, num_classes))
    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.gru.input_size:
            raise ValueError('Expected (batch,frames,features)')
        out, _ = self.gru(x)
        print('Helloe')
        return self.head(out[:, -1])
