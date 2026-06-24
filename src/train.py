"""
Training and cross-validation utilities for all Phase 1 models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# XGBoost cross-validation
# ──────────────────────────────────────────────────────────────

def cv_xgboost(model, X, y, n_splits=5, random_state=42):
    """5-fold stratified cross-validation for XGBoost."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        score = balanced_accuracy_score(y_val, preds)
        fold_scores.append(score)
        print(f"  Fold {fold+1}: balanced accuracy = {score:.4f}")

    mean, std = np.mean(fold_scores), np.std(fold_scores)
    print(f"\nXGBoost CV  →  {mean:.4f} ± {std:.4f}  (paper baseline: 0.631)")
    return fold_scores


# ──────────────────────────────────────────────────────────────
# PyTorch training loop (shared by CNN1D and TCN)
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    ba = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), ba


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    ba = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), ba, np.array(all_preds), np.array(all_labels)


def cv_pytorch(model_fn, dataset, y, n_splits=5, epochs=50,
               batch_size=16, lr=1e-3, random_state=42):
    """
    5-fold stratified cross-validation for PyTorch models (CNN1D / TCN).

    model_fn: callable with no args that returns a new model instance
    dataset : PyTorch Dataset (TemporalDataset or TabularDataset)
    y       : numpy array of integer labels (for stratification)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n── Fold {fold+1}/{n_splits} ──")

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  batch_size=batch_size, shuffle=False)

        model = model_fn().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # class weights to handle imbalance
        class_counts = np.bincount(y[train_idx])
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_ba, best_state = 0, None

        for epoch in range(epochs):
            tr_loss, tr_ba = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_ba, _, _ = eval_epoch(model, val_loader, criterion)
            scheduler.step()

            if val_ba > best_ba:
                best_ba = val_ba
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | "
                      f"train BA={tr_ba:.3f} loss={tr_loss:.4f} | "
                      f"val BA={val_ba:.3f} loss={val_loss:.4f}")

        fold_scores.append(best_ba)
        print(f"  Best val balanced accuracy: {best_ba:.4f}")

    mean, std = np.mean(fold_scores), np.std(fold_scores)
    print(f"\nModel CV  →  {mean:.4f} ± {std:.4f}")
    return fold_scores
