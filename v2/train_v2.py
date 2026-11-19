"""Train isolated-sign GRU. Legacy split is diagnostic only, NOT signer-independent."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .gru_model import SignGRU
from .temporal_dataset import TemporalDataset, split_indices

def run(data_path, output_dir, epochs=12, batch_size=32, seed=42, lr=0.001, groups_path=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = TemporalDataset(data_path)
    groups = np.load(groups_path, allow_pickle=False) if groups_path else None
    tr, val = split_indices(dataset.y.numpy(), seed=seed, groups=groups)
    train_ds = TemporalDataset(data_path, tr, dataset.classes)
    val_ds = TemporalDataset(data_path, val, dataset.classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = SignGRU(num_classes=len(dataset.classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -1.0
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        model.train(); total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(); logits = model(x); loss = loss_fn(logits, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += float(loss.item()) * len(y)
        model.eval(); truth=[]; pred=[]
        with torch.no_grad():
            for x, y in val_loader:
                pred.extend(model(x).argmax(-1).tolist()); truth.extend(y.tolist())
        acc = accuracy_score(truth, pred)
        history.append({'epoch': epoch+1, 'train_loss': total/len(train_ds), 'validation_accuracy': float(acc)})
        print(f'epoch {epoch+1}/{epochs}: loss={total/len(train_ds):.4f} val_accuracy={acc:.4f}', flush=True)
        if acc > best:
            best = float(acc)
            torch.save({'state_dict': model.state_dict(), 'classes': dataset.classes,
                        'input_size': 88, 'hidden_size': 96, 'layers': 2,
                        'seed': int(seed), 'evaluation_type': 'signer_independent' if groups_path else 'legacy_random_split_diagnostic'}, out/'best_gru.pt')
    checkpoint = torch.load(out/'best_gru.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.no_grad():
        logits = model(val_ds.x); predicted = logits.argmax(-1).numpy()
    report = {'best_validation_accuracy': float(best), 'final_checkpoint_accuracy': float(accuracy_score(val_ds.y, predicted)),
              'evaluation_type': checkpoint['evaluation_type'], 'train_samples': len(tr), 'validation_samples': len(val),
              'warning': None if groups_path else 'Random split of legacy data; not evidence of performance on unseen signers or sessions.',
              'history': history, 'classification_report': classification_report(val_ds.y, predicted, labels=list(range(len(dataset.classes))), target_names=dataset.classes, zero_division=0, output_dict=True),
              'confusion_matrix': confusion_matrix(val_ds.y, predicted, labels=list(range(len(dataset.classes)))).tolist()}
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    (out/'labels.json').write_text(json.dumps(dataset.classes, indent=2))
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='v2_data/legacy_features_v2.npz')
    p.add_argument('--out', default='v2_runs/gru_baseline')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--groups', default=None, help='Optional .npy array of signer IDs, aligned with dataset rows')
    a=p.parse_args()
    run(a.data, a.out, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed, groups_path=a.groups)
