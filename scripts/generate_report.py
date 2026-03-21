import argparse
import base64
import json
import os
from datetime import datetime

import pandas as pd

# ─────────────────────────────────────────────────────────────
# HELPER: encode a PNG chart to base64 so it embeds in HTML
# ─────────────────────────────────────────────────────────────

def encode_image(path):
  """
    Convert a PNG file to a base64 string.

    WHY: HTML reports must be single self-contained files.
    Embedding images as base64 means the report works anywhere
    without broken image links — critical for emailing reports
    or sharing with stakeholders who don't have file access.
    """

if not os.path.exists(path):
  return None
with open(path, "rb") as f:
  return base64.b64encode(f.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
# STAGE 5A: LOAD ALL ANALYSIS OUTPUTS
# ─────────────────────────────────────────────────────────────

def load_results(models_dir, validation_path):
    """
    Load all outputs from Stages 1-4."""

    print("\n📂 Stage 5A: Loading Analysis Outputs...")

    with open(f"{models_dir}/model_results.json") as f:
      models = json.load(f)
    print("   ✅ Loaded: model_results.json")

    #Load credit scores 
    scores_df = pd.read_csv(f"{models_dir}/credit_scores.csv")
    print(f"   ✅ Loaded: credit_scores.csv ({len(scores_df):,} rows)")

    return validation, models, scores_df
      
  def load_results(models_dir, validation_path):
    """
    Load all outputs from Stages 1-4."""
    print("\n📂 Stage 5A: Loading Analysis Outputs...")

    #Load validation report
    with open(validation_path) as f:
      validation = json.load(f)
    print("   ✅ Loaded: validation_report.json")

    #
    
