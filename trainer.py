from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from normalization import SequenceNormalizer, REPRESENTATION
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from database import load_database
from features import dataframe_to_xy


def split_data(df):
    X, y = dataframe_to_xy(df)
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError('Training needs at least two terms and two samples per term; collect more samples first.')
    n_test = int(np.ceil(len(y) * .2))
    if min(n_test, len(y)-n_test) < len(counts):
        raise ValueError('Too few samples for a stratified 80/20 evaluation; collect more samples per term.')
    return train_test_split(np.arange(len(df)), test_size=.2, random_state=42, stratify=y)


def evaluate_payload(payload, df, source):
    train_ids, test_ids = split_data(df)
    columns = payload['feature_columns']
    predictions = payload['model'].predict(df.iloc[test_ids][columns])
    truth = df.iloc[test_ids]['label']
    classes = payload['classes']
    hashes = pd.util.hash_pandas_object(df.drop(columns='label'), index=False)
    return {
        'accuracy': float(accuracy_score(truth, predictions)),
        'representation': payload.get('representation', 'raw_pixel_coordinates'),
        'correct': int(np.sum(predictions == truth)), 'test_samples': len(test_ids),
        'train_samples': len(train_ids), 'rows': len(df), 'terms': len(classes),
        'classes': classes, 'seed': 42, 'test_fraction': .2,
        'evaluation_type': 'legacy_stratified_random_holdout',
        'warning': 'Same-source samples only. No signer/session IDs: unseen-signer and live-camera accuracy are not established.',
        'database_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'train_indices': train_ids.tolist(), 'test_indices': test_ids.tolist(),
        'duplicate_feature_rows': int(hashes.duplicated().sum()),
        'identical_features_across_split': len(set(hashes.iloc[train_ids]) & set(hashes.iloc[test_ids])),
        'classification_report': classification_report(truth, predictions, labels=classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(truth, predictions, labels=classes).tolist(),
        'runtime': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
    }


def train_model(database_csv, model_path='models/asl_model.joblib'):
    database_csv, model_path = Path(database_csv), Path(model_path)
    df = load_database(database_csv)
    X, y = dataframe_to_xy(df)
    train_ids, _ = split_data(df)
    model = Pipeline([('normalize', SequenceNormalizer()), ('classifier', ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1))])
    model.fit(X.iloc[train_ids], y.iloc[train_ids])
    payload = {'model': model, 'feature_columns': list(X.columns), 'classes': model.classes_.tolist(), 'format_version': 2, 'representation': REPRESENTATION}
    report = evaluate_payload(payload, df, database_csv)
    report.update(database_csv=str(database_csv), model_path=str(model_path), estimator='ExtraTreesClassifier', n_estimators=300)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_suffix('.tmp')
    joblib.dump(payload, temp)
    temp.replace(model_path)
    model_path.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}. Run python app.py train.')
    # Only load trusted local model files (joblib uses pickle).
    payload = joblib.load(path)
    if not {'model', 'feature_columns', 'classes'} <= payload.keys():
        raise ValueError('Invalid model package; retrain from CSV.')
    return payload
