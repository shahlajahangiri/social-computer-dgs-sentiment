
# Notebook 2: Feature Selection + Engineering
# Goal: From 396 raw features to select the best ones + create new features

import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from src.dataset import load_tabular, get_xy

df, feature_cols, label_col = load_tabular()
X, y, feature_names, scaler = get_xy(scale=False)   # raw, unscaled

print(f"Starting with {len(feature_names)} features")

# STEP 0: Remove face features (not implemented in our system)
# The paper notes face tracking as a known limitation. Keeping face features would inflate results we cannot reproduce in Phase 2.
FACE_KEYWORDS = ["eye", "mouth", "brow", "lip", "nose", "face", "cheek",
                 "jaw", "smile", "neutral", "funnel", "roll", "pucker"]
face_mask = [not any(kw in f.lower() for kw in FACE_KEYWORDS) for f in feature_names]
X = X[:, face_mask]
feature_names = [f for f, keep in zip(feature_names, face_mask) if keep]
n_removed = sum(not k for k in face_mask)
print(f"Removed {n_removed} face features → {len(feature_names)} body/motion features remain")

# STEP 1: Remove near-zero-variance features / Features that barely change across all segments carry no information
from sklearn.feature_selection import VarianceThreshold

selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X)
kept_mask = selector_var.get_support()
feature_names_var = [f for f, k in zip(feature_names, kept_mask) if k]
print(f"After variance threshold: {X_var.shape[1]} features (removed {len(feature_names) - X_var.shape[1]})")

# STEP 2: ANOVA F-score / how well does each feature separate classes?
selector_f = SelectKBest(f_classif, k=min(150, X_var.shape[1]))
X_fsel = selector_f.fit_transform(X_var, y)
f_scores = selector_f.scores_
kept_f_mask = selector_f.get_support()
feature_names_f = [f for f, k in zip(feature_names_var, kept_f_mask) if k]
print(f"After ANOVA F-selection (top 150): {X_fsel.shape[1]} features")

# Plot top feature scores
top_idx = np.argsort(f_scores[kept_f_mask])[::-1][:20]
top_feats = [feature_names_f[i] for i in top_idx]
top_scores = [f_scores[kept_f_mask][i] for i in top_idx]

plt.figure(figsize=(10, 5))
plt.barh(top_feats[::-1], top_scores[::-1], color="#3498db")
plt.xlabel("ANOVA F-score")
plt.title("Top 20 Features by ANOVA F-score")
plt.tight_layout()
plt.savefig("../outputs/plots/feature_fscores.png", dpi=150)
plt.show()

# STEP 3: Mutual Information / captures non-linear relationships
selector_mi = SelectKBest(mutual_info_classif, k=min(100, X_fsel.shape[1]))
X_misel = selector_mi.fit_transform(X_fsel, y)
kept_mi_mask = selector_mi.get_support()
feature_names_final = [f for f, k in zip(feature_names_f, kept_mi_mask) if k]
print(f"After Mutual Information selection (top 100): {X_misel.shape[1]} features")

# STEP 4: Feature Engineering — create new temporal features /We add interaction terms between face and body features (body + face interaction matters in DGS paper finding!)

# Group features by body part (best effort naming heuristic)
face_feats = [f for f in feature_names_final
              if any(k in f.lower() for k in ["eye", "mouth", "brow", "lip", "nose", "face"])]
body_feats = [f for f in feature_names_final
              if any(k in f.lower() for k in ["shoulder", "elbow", "wrist", "hip", "hand"])]

print(f"\nFace-related features : {len(face_feats)}")
print(f"Body-related features : {len(body_feats)}")

# Create face-body ratio features (novel / not in original paper)
X_eng = X_misel.copy()
eng_names = list(feature_names_final)

if face_feats and body_feats:
    face_idx = [feature_names_final.index(f) for f in face_feats[:5]]
    body_idx = [feature_names_final.index(f) for f in body_feats[:5]]

    for fi, bi in zip(face_idx, body_idx):
        ratio = X_eng[:, fi] / (np.abs(X_eng[:, bi]) + 1e-6)
        X_eng = np.column_stack([X_eng, ratio])
        eng_names.append(f"ratio_{feature_names_final[fi]}_over_{feature_names_final[bi]}")

    print(f"After feature engineering: {X_eng.shape[1]} features")

# STEP 5: Scale and save processed dataset
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_eng)

# Build final dataframe
processed_df = pd.DataFrame(X_scaled, columns=eng_names)
processed_df["label"] = y
processed_df.to_csv("../outputs/processed_dataset.csv", index=False)

print(f"\n✓ Processed dataset saved: {X_scaled.shape}")
print(f"  → ../outputs/processed_dataset.csv")
print(f"\nLabel distribution in processed data:")
print(pd.Series(y).value_counts().rename({0:"negative", 1:"neutral", 2:"positive"}))
print("\nNext → run 03_baseline_xgboost.py")
