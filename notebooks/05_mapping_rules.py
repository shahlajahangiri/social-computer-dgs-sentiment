# Notebook 5 — Mapping Rules + Phase 2 Handoff
# Goal: Generate the "Mapping Rules" output from the Phase 1 workflow diagram.
# Mapping Rules = a structured table that tells Phase 2: "when this body part moves in this way then apply this sentiment (valence)"

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

# Load the processed dataset + trained XGBoost model
proc = pd.read_csv("../outputs/processed_dataset.csv")
feature_cols = [c for c in proc.columns if c != "label"]
X = proc[feature_cols].values.astype(np.float32)
y = proc["label"].values.astype(int)

xgb_model = joblib.load("../outputs/models/xgboost_final.pkl")
print("Models loaded.")

# SHAP Mapping Rules (already computed in notebook 3)
from src.evaluate import extract_mapping_rules_xgboost, plot_shap_summary

rules_df = extract_mapping_rules_xgboost(xgb_model, X, feature_cols, top_k=30)
print("\nFull Mapping Rules table:")
print(rules_df.to_string(index=False))

# Plot: which body part drives each sentiment?
def _extract_body_part(feat_name):
    feat = feat_name.lower()
    for part in ["eye", "mouth", "brow", "lip", "nose",
                 "shoulder", "elbow", "wrist", "hip", "hand",
                 "head", "spine", "face"]:
        if part in feat:
            return part
    return "other"

rules_df["body_part"] = rules_df["feature_name"].apply(_extract_body_part)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sentiments = ["negative", "neutral", "positive"]
shap_cols  = ["shap_negative", "shap_neutral", "shap_positive"]

for ax, sentiment, col in zip(axes, sentiments, shap_cols):
    top10 = rules_df.nlargest(10, col)
    ax.barh(top10["feature_name"].str[:30].values[::-1],
            top10[col].values[::-1],
            color={"negative":"#e74c3c","neutral":"#95a5a6","positive":"#2ecc71"}[sentiment])
    ax.set_title(f"Top features → '{sentiment}'")
    ax.set_xlabel("SHAP importance")

plt.suptitle("Mapping Rules: Feature → Sentiment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/plots/mapping_rules_by_sentiment.png", dpi=150)
plt.show()

# Generate the final Mapping Rules CSV for Phase 2

# Enrich with human-readable rule descriptions
def make_rule_text(row):
    feat = row["feature_name"]
    sent = row["dominant_sentiment"]
    imp  = row["global_importance"]
    return f"High '{feat}' → sentiment: {sent}  (importance={imp:.3f})"

rules_df["rule_description"] = rules_df.apply(make_rule_text, axis=1)

final_rules = rules_df[["rank", "feature_name", "body_part",
                          "dominant_sentiment", "global_importance",
                          "shap_negative", "shap_neutral", "shap_positive",
                          "rule_description"]]

final_rules.to_csv("../outputs/rules/mapping_rules_FINAL.csv", index=False)
print("\n✓ FINAL Mapping Rules saved:")
print("  → ../outputs/rules/mapping_rules_FINAL.csv")
print("\nThis file feeds directly into Phase 2 as 'Input Valence'.")

# Also export valence scores per segment (continuous -1 to +1)
valence_df = pd.read_csv("../outputs/rules/valence_scores_xgboost.csv")
print("\nSample valence scores for Phase 2:")
print(valence_df.head(10))

# Summary dashboard
print("\n" + "="*60)
print("  PHASE 1 COMPLETE — SUMMARY")
print("="*60)

comparison = pd.read_csv("../outputs/model_comparison.csv") \
    if os.path.exists("../outputs/model_comparison.csv") else pd.DataFrame()
if not comparison.empty:
    print(comparison.to_string(index=False))

print(f"""
Outputs produced:
  outputs/processed_dataset.csv          ← engineered features
  outputs/model_comparison.csv           ← all model scores
  outputs/models/xgboost_final.pkl       ← trained XGBoost
  outputs/models/cnn1d_final.pt          ← trained 1D CNN
  outputs/models/tcn_final.pt            ← trained TCN
  outputs/rules/mapping_rules_FINAL.csv  ← MAPPING RULES → Phase 2
  outputs/rules/valence_scores_xgboost.csv ← per-segment valence → Phase 2
  outputs/plots/                         ← all figures for the paper

PHASE 2 inputs:
  - mapping_rules_FINAL.csv  → used by the Emotional Post Processing Layer
  - valence_scores_xgboost.csv → "Input Valence" for each MMS segment
""")
