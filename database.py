from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
from features import feature_columns


LABEL_COL = "label"


def load_database(csv_path: str | Path) -> pd.DataFrame:
    """Load the ASL landmark database and remove accidental index columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Database not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    if LABEL_COL not in df.columns:
        raise ValueError(f"Database must contain a '{LABEL_COL}' column.")
    return validate_database(df)


def validate_same_schema(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Validate that a database update has the same feature columns."""
    incoming = incoming.loc[:, ~incoming.columns.str.contains(r"^Unnamed")]

    if LABEL_COL not in incoming.columns:
        raise ValueError(f"Update file must contain a '{LABEL_COL}' column.")

    existing_cols = list(existing.columns)
    incoming_cols = list(incoming.columns)

    missing = [c for c in existing_cols if c not in incoming_cols]
    extra = [c for c in incoming_cols if c not in existing_cols]

    if missing or extra:
        raise ValueError(
            "Update CSV schema does not match the current database.\n"
            f"Missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"Extra columns: {extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    return validate_database(incoming[existing_cols])


def backup_database(csv_path: str | Path, backup_dir: str | Path = "backups") -> Path:
    """Create a timestamped backup before modifying the database."""
    csv_path = Path(csv_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{csv_path.stem}_{stamp}.csv"
    shutil.copy2(csv_path, backup_path)
    return backup_path


def append_update(database_csv: str | Path, update_csv: str | Path, backup: bool = True) -> dict:
    """Append new ASL samples into the database CSV."""
    database_csv = Path(database_csv)
    update_csv = Path(update_csv)

    existing = load_database(database_csv)
    incoming = pd.read_csv(update_csv)
    incoming = validate_same_schema(existing, incoming)

    backup_path = backup_database(database_csv, database_csv.parent / "backups") if backup else None

    combined = pd.concat([existing, incoming], ignore_index=True)
    temp = database_csv.with_suffix(".tmp")
    combined.to_csv(temp, index=False)
    temp.replace(database_csv)

    return {
        "database": str(database_csv),
        "backup": str(backup_path) if backup_path else None,
        "added_rows": int(len(incoming)),
        "total_rows": int(len(combined)),
        "labels_added": sorted(map(str, incoming[LABEL_COL].unique().tolist())),
    }


def list_terms(database_csv: str | Path) -> pd.DataFrame:
    """Return label counts in the current database."""
    df = load_database(database_csv)
    return (
        df[LABEL_COL]
        .value_counts()
        .rename_axis("term")
        .reset_index(name="samples")
        .sort_values(["term"])
    )

def validate_database(df):
    expected = feature_columns()
    if set(df.columns) != set(expected + [LABEL_COL]):
        raise ValueError("Expected exactly 1260 named landmark columns plus label.")
    if df.empty or df[LABEL_COL].isna().any():
        raise ValueError("Database must have samples and nonempty labels.")
    df = df.loc[:, expected + [LABEL_COL]].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    if (df[LABEL_COL] == "").any():
        raise ValueError("Labels cannot be blank.")
    try:
        values = df[expected].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmarks must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Landmarks cannot contain NaN or infinity.")
    df[expected] = values
    return df
