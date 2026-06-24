# Goal: Replicate and *beat* the paper's XGBoost result (0.631 balanced accuracy).
# Why this first? Fast to train (seconds, not hours) / Gives us feature importance → first version of Mapping Rules / Sets the performance bar for the CNN to beat

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from src.models import build_xgboost
from src.train import cv_xgboost
from src.evaluate import print_report, plot_confusion_matrix, extract_mapping_rules_xgboost, export_valence_scores

# Load processed dataset
proc = pd.read_csv("../outputs/processed_dataset.csv")
feature_cols = [c for c in proc.columns if c != "label"]
X = proc[feature_cols].values.astype(np.float32)
y = proc["label"].values.astype(int)

print(f"Dataset: {X.shape}  |  Classes: {np.bincount(y)}")

#  Handle class imbalance with SMOTE (creates synthetic minority-class samples so the model doesn't just predict "neutral")
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE: {X_resampled.shape}  |  Classes: {np.bincount(y_resampled)}")

#Cross-validation (same protocol as the paper: stratified k-fold)
print("\n── 5-fold Cross Validation ──")
model = build_xgboost()
fold_scores = cv_xgboost(model, X_resampled, y_resampled)

# Train final model on ALL data (for rule extraction + Phase 2 valence export)
print("\n── Training final model on full dataset ──")
final_model = build_xgboost()
final_model.fit(
    X_resampled, y_resampled,
    eval_set=[(X, y)],
    verbose=False,
)

# Evaluate on original (non-resampled) data
preds = final_model.predict(X)
print_report(y, preds, "XGBoost")
plot_confusion_matrix(y, preds, "XGBoost")

# Extract Mapping Rules (SHAP)
print("\n── Extracting Mapping Rules via SHAP ──")
rules_df = extract_mapping_rules_xgboost(final_model, X, feature_cols, top_k=30)

print("\nTop 10 Mapping Rules:")
print(rules_df.head(10)[["rank","feature_name","dominant_sentiment","global_importance"]].to_string(index=False))

# Export valence scores for Phase 2
segment_ids = list(range(len(X)))   # replace with actual segment IDs if available
valence_df = export_valence_scores(final_model, X, segment_ids, model_name="xgboost")

print("\nSample valence scores (first 5 rows):")
print(valence_df.head())

# Save the trained model
import joblib
joblib.dump(final_model, "../outputs/models/xgboost_final.pkl")
print("\n✓ Model saved → ../outputs/models/xgboost_final.pkl")
print("\nNext → run 04_temporal_cnn_tcn.py  (the novel contribution!)")
