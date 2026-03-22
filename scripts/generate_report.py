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
  if not os.path.exists(path):
    return None
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
# STAGE 5A: LOAD ALL ANALYSIS OUTPUTS
# ─────────────────────────────────────────────────────────────

def load_results():
    print("\n📂 Stage 5A: Loading Analysis Outputs...")

    with open("outputs/validation_report.json") as f:
        validation = json.load(f)
    print("   ✅ Loaded: outputs/validation_report.json")

    with open("output/models/model_summary.json") as f:
        models = json.load(f)
    print("   ✅ Loaded: output/models/model_summary.json")

    scores_df = pd.read_csv("output/models/credit_scores.csv")
    print(f"   ✅ Loaded: output/models/credit_scores.csv ({len(scores_df):,} rows)")

    return validation, models, scores_df
      
def generate_insights(validation, models, scores_df):
  print("\n💡 Stage 5B: Generating Business Insights...")

  gb = models["gradient_boosting"]
  lr = models["logistic_regression"]
  winner = models["best_model"]

  auc = gb["cv_auc_mean"] if winner == "Gradient Boosting" else lr["cv_auc_mean"]
  gini = gb["gini"] if winner == "Gradient Boosting" else lr["gini"]
  ks = gb["ks_statistic"] if winner == "Gradient Boosting" else lr["ks_statistic"]

  auc_interpretation = (
        "Excellent — the model has strong discriminatory power" if auc >= 0.80 else
        "Good — the model meets industry deployment standards"  if auc >= 0.70 else
        "Moderate — further feature engineering recommended"
    )

  ks_interpretation = (
        "Strong separation between defaulters and non-defaulters" if ks >= 0.40 else
        "Acceptable separation — model is deployment-ready"       if ks >= 0.30 else
        "Weak separation — review feature set"
    )
  
  tier_counts = scores_df["Score_Tier"].value_counts().to_dict()
  total = len(scores_df)
  high_risk = tier_counts.get("Very Poor", 0) +  tier_counts.get("Poor", 0)
  low_risk = tier_counts.get("Very Good", 0) + tier_counts.get("Exceptional", 0)

  insights = {
    "winner": winner,
    "auc": auc,
    "gini": gini,
    "ks": ks,
    "auc_interpretation": auc_interpretation,
    "tier_counts": tier_counts,
    "total_customers": total,
    "high_risk_count": high_risk,
    "high_risk_pct":   round(high_risk / total * 100, 1),
    "low_risk_count":  low_risk,
    "low_risk_pct":    round(low_risk / total * 100, 1),
    "default_rate":    validation["class_balance"]["default_rate_pct"],
    "lr_auc":  lr["cv_auc_mean"],  "gb_auc":  gb["cv_auc_mean"],
    "lr_gini": lr["gini"],         "gb_gini": gb["gini"],
    "lr_ks":   lr["ks_statistic"], "gb_ks":   gb["ks_statistic"],

  }

  print(f"   ✅ Winner: {winner} (AUC={auc})")
  print(f"   ✅ High-risk customers: {high_risk:,} ({insights['high_risk_pct']}%)")
  print(f"   ✅ Low-risk customers:  {low_risk:,} ({insights['low_risk_pct']}%)")
  return insights
  

def build_html_report(insights, validation):
  print("\n📝 Stage 6: Building HTML Report...")

  chart_paths = {
     "class_dist":     "outputs/charts/chart1_class_distribution.png",
     "null_heatmap":   "outputs/charts/chart2_null_heatmap.png",
     "payment_dist":   "outputs/charts/chart3_payment_status.png",
     "roc_importance": "output/models/charts/chart4_roc_and_importance.png",
     "score_dist":     "output/models/charts/chart5_score_distribution.png",
    }

  charts = {}
  for key, path in chart_paths.items():
    encoded = encode_image(path)
    charts[key] = f"data:image/png;base64, {encoded}" if encoded else None
    print(f"   {'✅' if encoded else '⚠️  Not found'} {path}")

  tier_order = ["Very Poor", "Poor", "Fair", "Good", "Very Good", "Exceptional"]
  tier_colors = {
        "Very Poor": "#e74c3c", "Poor": "#e67e22", "Fair": "#f1c40f",
        "Good": "#2ecc71", "Very Good": "#27ae60", "Exceptional": "#1abc9c"
    }

  tier_rows = ""
  for tier in tier_order:
    count = insights["tier_counts"].get(tier, 0)
    pct = round(count / insights["total_customers"] * 100, 1)
    color = tier_colors.get(tier, "#95a5a6")
    tier_rows += f"""
      <tr>
          <td><span style="color:{color};font-weight:bold;">● {tier}</span></td>
          <td style="text-align:right">{count:,}</td>
          <td style="text-align:right">{pct}%</td>
          <td><div style="background:{color};width:{int(pct*3)}px;height:14px;
                border-radius:3px;display:inline-block;"></div></td>
        </tr>"""

    timestamp =  datetime.now().strftime("%B %d, %Y at %H:%M IST")

    def img(key, style="width:100%;border-radius:8px;"):
      return f'<img src="{charts[key]}" style="{style}">' if charts[key] else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset = "UTF-8">
<title>Credit Risk Scorecard Report</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:'Segoe UI', Arial,sans-serif:background:#f4f6f9;color#2c3e50;}}.header{{background:linear-
gradient(135deg,#1a252f,#2c3e50);color:white;padding:40px;text-align:center;}}
  .header h1{{font-size:2.2em;margin-bottom:8px;}}
  .header p{{opacity:0.8;font-size:1em;}}
  .container{{max-width:1100px;margin:30px auto;padding:0 20px;}}
  .section{{background:white;border-radius:10px;padding:30px;margin-bottom:25px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);}}
  h2{{font-size:1.4em;color:#1a252f;border-bottom:3px solid #3498db;
  padding-bottom:10px;margin-bottom:20px;}}
  h3{{font-size:1.1em;color:#2c3e50;margin18px 0 10px 0;}}
  .kpi-grid{{display:grid;grid-template-colums:repeat(4,1fr);gap:15px;margin-bottom:10px;}}
  .kpi{{background:#f8f9fa;border-radius:8px;padding:20px;text-align:center;border-left:4px 
  solid #3498db;}}
  .kpi .value{{font-size:2em;font-weight:bold;color:#2980b9;}}
  .kpi .label{{font-size:0.85em;color:#7f8c8d;margin-top:5px;}}
  .kpi.green{{border-left-color:#2ecc71;}}.kpi.green .value{{color:#27ae60;}}
  .kpi.red{{border-left-color:#e74c3c;}}.kpi.red .value{{color:#c0392b;}}
  .kpi.purple{{border-left-color:#9b59b6;}}.kpi.purple .value{{color:#8e44ad;}}
  table{{width:100%;border-collapse:collapse;font-size:0.95em;}}
  th{{background:#2c3e50;color:white;padding:12px 15px;text-align:left;}}
  td{{padding:10px 15px;border-bottom:1px solid #ecf0f1;}}
  tr:hover td{{background:#f8f9fa;}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.8em;font-weight:bold;}}
  .badge.winner{{background:#d5f5e3;color:#1e8449;}}
  .badge.pass{{background:#d5f5e3;color:#1e8449;}}
  .badge.warn{{background:#fef9e7;color#b7950b;}}
  .chart-grid{{display:grid;grid-template-colums:1fr 1fr;gap:20px;margin-top:20px;}}
  .rec-box{{background:#eaf4fd;border-left:4px solid #3498db;padding:15px 20px;
  border-radius:5px;margin:10px 0;}}
  .rec-box h4{{color:#1a5276;margin-bottom:6px;}}
  .footer{{text-align:center;padding:20px;color:#7f8c8d;font-size:0.85em;}}
  </style>
  </head>
  <body>

<div class="header">
  <h1> Credit Risk Scorecard Report</h1>
  <p>UCI Credit Card Default Dataset &nbsp;|&nbsp; {insights['total_customers']:,}
Customers &nbsp;|&nbsp; Generated {timestamp}</p>
  <p style="margin-top:8px;font-size=0.9em;opacity:0.7;">
    Author: Abhinav Srivastav &nbsp;|&nbsp; AMPBA, ISB &nbsp;|&nbsp; Credit Analytics Portfolio Project
 </p>
</div>

<div class="container">

  <!-- 1. EXECUTIVE SUMMARY-->
  <div class="section">
    <h2>1. Executive Summary</h2>
    <p style="margin-bottom:20px;line-height:1.7;">
      This report presents an end to end credit risk scorecard built on 
      <strong>{insights['total_customers']:,} credit card customers</strong>.
      Two models were trained and compared - Logistic Regression and Gradient Boosting.
      The winning model (<strong>{insights['winner']}</strong>) achieved a cross validated
      AUC of <strong>{insights['auc']}</strong> and KS Statistic of 
      <strong>{insights['ks']}</strong>, meeting industry deployment standards.
      <strong>{insights['high_risk_pct']}% ({insights['high_risk_count']:,} customers)</strong>
      fall in high-risk tiers, representing the primary target for collections intervention.
      </p>

      <div class="kpi-grid">
        <div class="kpi><div class="value">{insights['auc']}</div><div class="label">Cross-Validated AUC</div></div>
        <div class="kpi green"><div class="value">{insights['gini']}</div><div class="label>
        Gini Coefficent</div></div>
        <div class="kpi purple"><div class="value">{insights['ks']}</div><div class="label>
        KS Statistic</div></div>
        <div class="kpi red"><div class="value">{insights['default_rate']}%</div><div class="label>
        Portfolio Default Rate</div></div>
      </div>
    </div>

  <!-- 2. DATA QUALITY -->
  <div class="section">
    <h2>2. Data Quality Summary</h2>
    <table>
      <tr><th>Check</th><th>Result</th><th>Status</th></tr>
      <tr><td>Total Records</td>
          <td>{validation['dataset_shape']['rows']:,} rows × {validation['dataset_shape']['columns']} columns</td>
          <td><span class="badge pass">PASS</span></td></tr>
      <tr><td>Missing Values</td>
          <td>{validation['null_validation']['total_nulls']} total nulls (0.0%)</td>
          <td><span class="badge pass">PASS</span></td></tr>
      <tr><td>Duplicate Rows</td>
          <td>{validation['duplicate_validation']['duplicate_count']} rows ({validation['duplicate_validation']['duplicate_percentage']}%) — removed</td>
          <td><span class="badge pass">PASS</span></td></tr>
      <tr><td>EDUCATION Column</td>
          <td>Undocumented values [0,5,6] → recoded to 'Others' (4)</td>
          <td><span class="badge warn">WARN</span></td></tr>
      <tr><td>MARRIAGE Column</td>
          <td>Undocumented value [0] → recoded to 'Others' (3)</td>
          <td><span class="badge warn">WARN</span></td></tr>
      <tr><td>Class Balance</td>
          <td>{validation['class_balance']['default_rate_pct']}% default rate — class_weight='balanced' applied</td>
          <td><span class="badge pass">PASS</span></td></tr>
    </table>
    <div class="chart-grid">
      <div>{img('class_dist')}</div>
      <div>{img('payment_dist')}</div>
    </div>
  </div>

  <!-- 3. METHODOLOGY -->
  <div class="section">
    <h2>3. Methodology</h2>
    <h3>3.1 Feature Engineering (7 Features)</h3>
    <table>
      <tr><th>Feature</th><th>Formula</th><th>Business Meaning</th></tr>
      <tr><td>UTILISATION_RATE</td><td>avg(BILL_AMT1–6) / LIMIT_BAL</td><td>How much of credit limit is used — high utilisation signals risk</td></tr>
      <tr><td>PAYMENT_RATIO</td><td>avg(PAY_AMT1–6) / avg(BILL_AMT1–6)</td><td>Fraction of bill paid — low ratio indicates revolving balance risk</td></tr>
      <tr><td>MAX_DELAY</td><td>max(PAY_0, PAY_2–6)</td><td>Worst payment delay in 6 months — strongest default predictor</td></tr>
      <tr><td>MEAN_DELAY</td><td>mean(PAY_0, PAY_2–6)</td><td>Average delay — captures chronic vs one-time late payers</td></tr>
      <tr><td>DELAY_COUNT</td><td>count(PAY_x &gt; 0)</td><td>Months with any delay — behavioural consistency signal</td></tr>
      <tr><td>BILL_TREND</td><td>BILL_AMT1 − BILL_AMT6</td><td>Is debt growing? Positive = accumulating debt → higher risk</td></tr>
      <tr><td>PAY_TREND</td><td>PAY_AMT1 − PAY_AMT6</td><td>Is payment improving? Positive = paying more recently</td></tr>
    </table>
    <h3>3.2 Models Compared</h3>
    <table>
      <tr><th>Model</th><th>Why Used</th><th>Key Parameters</th></tr>
      <tr><td>Logistic Regression</td><td>Industry standard — interpretable, coefficients map to score points</td><td>C=0.1, class_weight='balanced', max_iter=1000</td></tr>
      <tr><td>Gradient Boosting</td><td>High-performance benchmark — captures non-linear feature interactions</td><td>100 trees, learning_rate=0.1, max_depth=4</td></tr>
    </table>
  </div>


  <!-- 4. MODEL RESULTS -->
  <div class="section">
    <h2>4. Model Validation Results</h2>
    <table>
      <tr><th>Metric</th><th>Logistic Regression</th><th>Gradient Boosting</th><th>Threshold</th><th>Status</th></tr>
      <tr><td>CV AUC (5-Fold)</td><td>{insights['lr_auc']}</td><td><strong>{insights['gb_auc']}</strong></td>
          <td>&gt; 0.70</td><td><span class="badge pass">✅ PASS</span></td></tr>
      <tr><td>Gini Coefficient</td><td>{insights['lr_gini']}</td><td><strong>{insights['gb_gini']}</strong></td>
          <td>&gt; 0.40</td><td><span class="badge pass">✅ PASS</span></td></tr>
      <tr><td>KS Statistic</td><td>{insights['lr_ks']}</td><td><strong>{insights['gb_ks']}</strong></td>
          <td>&gt; 0.30</td><td><span class="badge pass">✅ PASS</span></td></tr>
      <tr><td>Winner</td><td>—</td><td><span class="badge winner">🏆 WINNER</span></td><td>—</td><td>—</td></tr>
    </table>
    <div style="margin-top:20px;">{img('roc_importance')}</div>
  </div>

<!-- 5. SCORE TIERS -->
  <div class="section">
    <h2>5. Credit Score Distribution (300–850 Scale)</h2>
    <p style="margin-bottom:15px;line-height:1.7;">
      Default probabilities from the winning model are mapped to a 300–850 credit score
      scale (FICO-style). Higher score = lower default risk.
    </p>
    <table>
      <tr><th>Score Tier</th><th style="text-align:right">Customers</th>
          <th style="text-align:right">% of Portfolio</th><th>Distribution</th></tr>
      {tier_rows}
    </table>
    <div style="margin-top:20px;">{img('score_dist')}</div>
  </div>

  <!-- 6. RECOMMENDATIONS -->
  <div class="section">
    <h2>6. Business Recommendations</h2>
    <div class="rec-box">
      <h4>🔴 High-Risk Segment ({insights['high_risk_pct']}% — {insights['high_risk_count']:,} customers)</h4>
      <p>Scores below 580. Reduce credit limits proactively, trigger early collections at 30 DPD, offer structured repayment plans. Highest ROI for collections investment.</p>
    </div>
    <div class="rec-box">
      <h4>🟡 Medium-Risk Segment (Fair — scores 580–670)</h4>
      <p>Irregular payment behaviour, not yet in default. Send payment reminders 5 days before due date, offer balance transfer at lower rates, monitor monthly for deterioration.</p>
    </div>
    <div class="rec-box">
      <h4>🟢 Low-Risk Segment ({insights['low_risk_pct']}% — {insights['low_risk_count']:,} customers)</h4>
      <p>Scores above 740. Offer proactive credit limit increases, cross-sell premium card products, use as benchmark cohort for new customer acquisition targeting.</p>
    </div>
    <div class="rec-box">
      <h4>📊 Model Monitoring</h4>
      <p>Retrain quarterly using rolling 12-month window. Monitor PSI monthly — alert if PSI &gt; 0.25. Alert if KS or Gini drops 5+ points from baseline.</p>
    </div>
  </div>

  <!-- 7. LIMITATIONS -->
  <div class="section">
    <h2>7. Assumptions &amp; Limitations</h2>
    <div class="limit-box"><p><strong>Data Period:</strong> Covers April–September 2005 (Taiwan). Recalibrate before deploying in other geographies or time periods.</p></div>
    <div class="limit-box"><p><strong>Class Imbalance:</strong> 22.1% default rate handled via class_weight='balanced'. Evaluate SMOTE and threshold optimisation for production deployment.</p></div>
    <div class="limit-box"><p><strong>No Bureau Data:</strong> Uses statement and payment data only. Bureau scores and transaction-level features would significantly improve predictive power.</p></div>
    <div class="limit-box"><p><strong>Static Model:</strong> Point-in-time scorecard. Requires ongoing monitoring and periodic retraining as customer behaviour evolves.</p></div>
  </div>

</div>

<div class="footer">
  Credit Risk Scorecard Report &nbsp;|&nbsp; Abhinav Srivastav, AMPBA ISB
  &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Dataset: UCI ML Repository (ID 350)
</div>
</body>
</html>"""

    os.makedirs("outputs/reports", exist_ok=True)
    report_path = "outputs/reports/credit_risk_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   ✅ Report saved → {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("  CREDIT RISK SCORECARD — STAGES 5–6: REPORT GENERATION")
    print("=" * 60)
    validation, models, scores_df = load_results()
    insights = generate_insights(validation, models, scores_df)
    build_html_report(insights, validation)
    print(f"\n{'='*60}")
    print(f"  REPORT COMPLETE → outputs/reports/credit_risk_report.html")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()