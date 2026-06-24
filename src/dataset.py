"""
Data loading and preprocessing for DGS-Fabeln-1-SE dataset.

Expected files in ../data/:
  - DGS-Fabeln-1-SE-Labels.csv        (segment_id + sentiment label)
  - DGS-Fabeln-1-SE-MotionFeatures.csv (segment_id + 396 motion features)
  - <FairyTale>-MediaPipe.csv          (one per fairy tale, frame-level data)

Download from: https://zenodo.org/records/18879038
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]


# ──────────────────────────────────────────────────────────────
# 1. Tabular data (for XGBoost + baseline models)
# ──────────────────────────────────────────────────────────────

def load_tabular(data_dir=DATA_DIR):
    """Load and merge Labels + MotionFeatures into one DataFrame."""
    # ── find files (names may vary slightly on download)
    labels_file = _find_file(data_dir, "*Labels*")
    features_file = _find_file(data_dir, "*MotionFeatures*")

    labels_df = pd.read_csv(labels_file)
    features_df = pd.read_csv(features_file)

    print(f"Labels shape    : {labels_df.shape}")
    print(f"Features shape  : {features_df.shape}")
    print(f"Label columns   : {list(labels_df.columns)}")
    print(f"Label counts    :\n{labels_df.iloc[:, -1].value_counts()}\n")

    # ── identify key columns (flexible naming)
    label_col = _find_label_col(labels_df)
    id_col = _find_id_col(labels_df)

    # merge on Story+id composite key if both files share a Story column
    story_col = "Story" if "Story" in features_df.columns and "Story" in labels_df.columns else None
    merge_keys = [story_col, id_col] if story_col else [id_col]
    label_cols_to_keep = merge_keys + [label_col]
    merged = features_df.merge(labels_df[label_cols_to_keep], on=merge_keys, how="inner")
    print(f"Merged shape    : {merged.shape}")

    # ── encode labels to 0/1/2  ('multi' = annotators disagreed → drop)
    merged["label_enc"] = merged[label_col].str.lower().map(LABEL_MAP)
    n_before = len(merged)
    merged = merged.dropna(subset=["label_enc"])
    merged["label_enc"] = merged["label_enc"].astype(int)
    n_dropped = n_before - len(merged)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with ambiguous labels (e.g. 'multi')")

    non_feature = {id_col, label_col, "label_enc", "Story"}
    feature_cols = [c for c in merged.columns if c not in non_feature]
    return merged, feature_cols, "label_enc"


def get_xy(data_dir=DATA_DIR, scale=True):
    """Return (X, y, feature_names, scaler) ready for sklearn / XGBoost."""
    df, feature_cols, label_col = load_tabular(data_dir)

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    # ── replace NaN with column median (some features may be missing)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    print(f"X shape: {X.shape}  |  y distribution: {np.bincount(y)}")
    return X, y, feature_cols, scaler


# ──────────────────────────────────────────────────────────────
# 2. Temporal data (for 1D CNN / TCN)
# ──────────────────────────────────────────────────────────────

def load_temporal(data_dir=DATA_DIR, max_len=300):
    """
    Load per-frame MediaPipe CSVs and return padded sequences.

    Returns:
        sequences : np.ndarray  shape (N, max_len, n_features)
        labels    : np.ndarray  shape (N,)
        seg_ids   : list of segment IDs
    """
    labels_file = _find_file(data_dir, "*Labels*")
    labels_df = pd.read_csv(labels_file)
    label_col = _find_label_col(labels_df)
    id_col = _find_id_col(labels_df)
    labels_df["label_enc"] = labels_df[label_col].str.lower().map(LABEL_MAP)

    mediapipe_files = glob.glob(os.path.join(data_dir, "*MediaPipe*.csv"))
    if not mediapipe_files:
        raise FileNotFoundError(
            "No MediaPipe CSV files found in data/. "
            "Download the raw MediaPipe files from zenodo.org/records/18879038"
        )

    all_seqs, all_labels, all_ids = [], [], []

    for mp_file in sorted(mediapipe_files):
        print(f"Loading {os.path.basename(mp_file)} ...")
        mp_df = pd.read_csv(mp_file)

        # Identify segment column and frame column
        frame_col = _find_col_containing(mp_df, ["frame", "timestamp", "time"])
        story_col = "Story" if "Story" in mp_df.columns else None
        non_feature = {frame_col, "Story"} if story_col else {frame_col}
        feature_cols = [c for c in mp_df.columns if c not in non_feature | {id_col}]

        group_keys = [story_col, id_col] if story_col and story_col in mp_df.columns else [id_col]
        for group_key, group in mp_df.groupby(group_keys):
            if isinstance(group_key, tuple):
                story_val, seg_id = group_key
            else:
                story_val, seg_id = None, group_key

            if story_val is not None:
                label_row = labels_df[(labels_df[id_col] == seg_id) & (labels_df["Story"] == story_val)]
            else:
                label_row = labels_df[labels_df[id_col] == seg_id]
            if label_row.empty:
                continue
            label = label_row["label_enc"].values[0]
            if pd.isna(label):
                continue

            seq = group[feature_cols].values.astype(np.float32)
            # truncate or pad to max_len
            if len(seq) > max_len:
                seq = seq[:max_len]
            elif len(seq) < max_len:
                pad = np.zeros((max_len - len(seq), seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])

            all_seqs.append(seq)
            all_labels.append(int(label))
            all_ids.append(seg_id)

    sequences = np.stack(all_seqs)   # (N, max_len, n_features)
    labels = np.array(all_labels)
    print(f"\nTemporal dataset: {sequences.shape}, label dist: {np.bincount(labels)}")
    return sequences, labels, all_ids


# ──────────────────────────────────────────────────────────────
# 3. PyTorch Dataset wrappers
# ──────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TemporalDataset(Dataset):
    """
    sequences shape: (N, time_steps, n_features)
    For 1D CNN / TCN we need (N, n_features, time_steps)  ← channels first
    """
    def __init__(self, sequences, labels):
        # transpose: (N, T, C) → (N, C, T)
        self.X = torch.tensor(sequences.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {directory}.\n"
            f"Please download the dataset from https://zenodo.org/records/18879038 "
            f"and place all CSV files in the data/ folder."
        )
    return matches[0]


def _find_label_col(df):
    # Prefer Aggregated sentiment column (cleanest: neg/neutral/pos/multi)
    for col in df.columns:
        if "aggregated" in col.lower() and "sentiment" in col.lower():
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in ["label", "sentiment", "valence", "class"]):
            return col
    return df.columns[-1]


def _find_id_col(df):
    for col in df.columns:
        if any(kw in col.lower() for kw in ["id", "segment", "seg", "index"]):
            return col
    return df.columns[0]  # fallback: first column


def _find_col_containing(df, keywords):
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    return df.columns[0]
