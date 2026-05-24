"""
app.py
Interactive Credit Risk Scorecard Demo — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="💳",
    layout="wide"
)

# ── LOAD & TRAIN MODEL (cached so it only runs once) ─────────────────────────
@st.cache_resource(show_spinner="Training model on UCI dataset...")
def load_and_train():
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    col_map = {
        "X1":"LIMIT_BAL","X2":"SEX","X3":"EDUCATION","X4":"MARRIAGE","X5":"AGE",
        "X6":"PAY_0","X7":"PAY_2","X8":"PAY_3","X9":"PAY_4","X10":"PAY_5","X11":"PAY_6",
        "X12":"BILL_AMT1","X13":"BILL_AMT2","X14":"BILL_AMT3",
        "X15":"BILL_AMT4","X16":"BILL_AMT5","X17":"BILL_AMT6",
        "X18":"PAY_AMT1","X19":"PAY_AMT2","X20":"PAY_AMT3",
        "X21":"PAY_AMT4","X22":"PAY_AMT5","X23":"PAY_AMT6"
    }
    X = X.rename(columns=col_map)
    df = pd.concat([X, y.rename(columns={"Y": "DEFAULT"})], axis=1)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
    df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering
    bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    pay_cols  = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    delay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

    df["AVG_BILL_AMT"]    = df[bill_cols].mean(axis=1)
    df["AVG_PAY_AMT"]     = df[pay_cols].mean(axis=1)
    df["UTILISATION_RATE"] = (df["AVG_BILL_AMT"] / df["LIMIT_BAL"]).clip(0, 1)
    df["PAYMENT_RATIO"]   = (df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"] + 1)).clip(0, 5)
    df["MAX_DELAY"]       = df[delay_cols].max(axis=1)
    df["MEAN_DELAY"]      = df[delay_cols].mean(axis=1)
    df["DELAY_COUNT"]     = (df[delay_cols] > 0).sum(axis=1)
    df["BILL_TREND"]      = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["PAY_TREND"]       = df["PAY_AMT1"]  - df["PAY_AMT6"]

    feature_cols = [
        "LIMIT_BAL","AGE","EDUCATION","MARRIAGE","SEX",
        "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
        "UTILISATION_RATE","PAYMENT_RATIO","MAX_DELAY",
        "MEAN_DELAY","DELAY_COUNT","BILL_TREND","PAY_TREND"
    ]

    X_feat = df[feature_cols].fillna(0)
    y_feat = df["DEFAULT"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    model = LogisticRegression(
        class_weight="balanced", random_state=42,
        max_iter=1000, C=0.1
    )
    model.fit(X_scaled, y_feat)

    return model, scaler, feature_cols


def prob_to_score(prob):
    """Map default probability to FICO-aligned 300-850 score."""
    score = int(round(850 - prob * 550))
    return max(300, min(850, score))


def get_tier(score):
    """Return tier label and colour for a given score."""
    if score >= 800:   return "Exceptional",  "#27ae60"
    elif score >= 740: return "Very Good",     "#2ecc71"
    elif score >= 670: return "Good",          "#f39c12"
    elif score >= 580: return "Fair",          "#e67e22"
    elif score >= 500: return "Poor",          "#e74c3c"
    else:              return "Very Poor",     "#c0392b"


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model, scaler, feature_cols = load_and_train()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("💳 Credit Risk Scorecard")
st.markdown(
    "**Interactive demo** — Enter customer details to generate a FICO-aligned "
    "credit score (300–850). Trained on UCI Credit Card Default dataset (30,000 customers)."
)
st.markdown("---")

# ── SIDEBAR INPUTS ────────────────────────────────────────────────────────────
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details below:")

limit_bal  = st.sidebar.number_input("Credit Limit (NT$)",       min_value=10000,  max_value=1000000, value=200000, step=10000)
age        = st.sidebar.slider("Age",                             min_value=18,     max_value=75,      value=35)
sex        = st.sidebar.selectbox("Gender",                       ["Male (1)", "Female (2)"])
education  = st.sidebar.selectbox("Education",                    ["Graduate School (1)", "University (2)", "High School (3)", "Others (4)"])
marriage   = st.sidebar.selectbox("Marital Status",               ["Married (1)", "Single (2)", "Others (3)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Payment History (last 6 months)")
st.sidebar.markdown("*-1 = Paid duly, 0 = Minimum paid, 1–9 = Months delayed*")

pay_0 = st.sidebar.slider("Most Recent Month (PAY_0)", -2, 9, 0)
pay_2 = st.sidebar.slider("2 Months Ago (PAY_2)",      -2, 9, 0)
pay_3 = st.sidebar.slider("3 Months Ago (PAY_3)",      -2, 9, 0)
pay_4 = st.sidebar.slider("4 Months Ago (PAY_4)",      -2, 9, 0)
pay_5 = st.sidebar.slider("5 Months Ago (PAY_5)",      -2, 9, 0)
pay_6 = st.sidebar.slider("6 Months Ago (PAY_6)",      -2, 9, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Financials")
avg_bill = st.sidebar.number_input("Average Monthly Bill (NT$)",    min_value=0, max_value=500000, value=50000, step=1000)
avg_pay  = st.sidebar.number_input("Average Monthly Payment (NT$)", min_value=0, max_value=500000, value=5000,  step=1000)
bill_trend = st.sidebar.number_input("Bill Trend (Recent - Oldest NT$)", min_value=-200000, max_value=200000, value=0, step=1000)
pay_trend  = st.sidebar.number_input("Pay Trend (Recent - Oldest NT$)",  min_value=-200000, max_value=200000, value=0, step=1000)

# ── COMPUTE ───────────────────────────────────────────────────────────────────
sex_val  = 1 if "Male" in sex else 2
edu_val  = int(education.split("(")[1].replace(")", ""))
mar_val  = int(marriage.split("(")[1].replace(")", ""))

delay_vals = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
utilisation = min(1.0, max(0.0, avg_bill / (limit_bal + 1)))
pay_ratio   = min(5.0, max(0.0, avg_pay / (avg_bill + 1)))
max_delay   = max(delay_vals)
mean_delay  = np.mean(delay_vals)
delay_count = sum(1 for d in delay_vals if d > 0)

input_data = pd.DataFrame([{
    "LIMIT_BAL":       limit_bal,
    "AGE":             age,
    "EDUCATION":       edu_val,
    "MARRIAGE":        mar_val,
    "SEX":             sex_val,
    "PAY_0":           pay_0,
    "PAY_2":           pay_2,
    "PAY_3":           pay_3,
    "PAY_4":           pay_4,
    "PAY_5":           pay_5,
    "PAY_6":           pay_6,
    "UTILISATION_RATE": utilisation,
    "PAYMENT_RATIO":   pay_ratio,
    "MAX_DELAY":       max_delay,
    "MEAN_DELAY":      mean_delay,
    "DELAY_COUNT":     delay_count,
    "BILL_TREND":      bill_trend,
    "PAY_TREND":       pay_trend,
}])

X_input  = scaler.transform(input_data[feature_cols])
prob     = model.predict_proba(X_input)[0][1]
score    = prob_to_score(prob)
tier, color = get_tier(score)

# ── RESULTS ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Credit Score")
    st.markdown(
        f"<h1 style='color:{color}; font-size:72px; margin:0'>{score}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='background:{color}; color:white; padding:4px 14px; "
        f"border-radius:12px; font-size:16px; font-weight:bold'>{tier}</span>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### Default Probability")
    st.markdown(
        f"<h1 style='color:{'#e74c3c' if prob > 0.5 else '#27ae60'}; font-size:72px; margin:0'>"
        f"{prob*100:.1f}%</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Probability of default in next month")

with col3:
    st.markdown("### Risk Decision")
    if score >= 670:
        decision, dec_color, icon = "APPROVE", "#27ae60", "✅"
    elif score >= 580:
        decision, dec_color, icon = "REVIEW",  "#f39c12", "⚠️"
    else:
        decision, dec_color, icon = "DECLINE", "#e74c3c", "❌"

    st.markdown(
        f"<h1 style='color:{dec_color}; font-size:52px; margin:0'>{icon} {decision}</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Based on FICO-aligned thresholds")

st.markdown("---")

# ── SCORE GAUGE ───────────────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("### Score Breakdown — FICO Scale")
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.set_xlim(300, 850)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bands = [
        (300, 500, "#c0392b", "Very Poor"),
        (500, 580, "#e74c3c", "Poor"),
        (580, 670, "#e67e22", "Fair"),
        (670, 740, "#f39c12", "Good"),
        (740, 800, "#2ecc71", "Very Good"),
        (800, 850, "#27ae60", "Exceptional"),
    ]
    for start, end, clr, label in bands:
        ax.barh(0, end - start, left=start, height=0.5, color=clr, alpha=0.85)
        ax.text((start + end) / 2, -0.15, label, ha="center", va="top", fontsize=8)

    ax.annotate(
        f"{score}", xy=(score, 0.25), xytext=(score, 0.85),
        fontsize=13, fontweight="bold", ha="center", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=2)
    )
    st.pyplot(fig)
    plt.close()

with col_b:
    st.markdown("### Score Tiers")
    tiers_data = {
        "Tier": ["Exceptional", "Very Good", "Good", "Fair", "Poor", "Very Poor"],
        "Range": ["800–850", "740–800", "670–740", "580–670", "500–580", "300–500"],
        "Decision": ["✅ Approve", "✅ Approve", "✅ Approve", "⚠️ Review", "❌ Decline", "❌ Decline"]
    }
    st.dataframe(pd.DataFrame(tiers_data), hide_index=True, use_container_width=True)

st.markdown("---")

# ── KEY RISK FACTORS ─────────────────────────────────────────────────────────
st.markdown("### Key Risk Factors")

def risk_label(value, high_thresh, mid_thresh, reverse=False):
    """Return emoji risk label."""
    if reverse:
        return "🟢 Good" if value > high_thresh else "🔴 Low" if value < mid_thresh else "🟡 Medium"
    return "🔴 High Risk" if value >= high_thresh else "🟡 Watch" if value >= mid_thresh else "🟢 Low Risk"

factors_df = pd.DataFrame([
    {
        "Risk Factor":    "Credit Utilisation Rate",
        "Your Value":     f"{utilisation*100:.1f}%",
        "Risk Signal":    "🔴 High Risk" if utilisation > 0.7 else "🟡 Watch" if utilisation > 0.4 else "🟢 Low Risk",
        "What It Means":  "High utilisation = large outstanding balances relative to limit"
    },
    {
        "Risk Factor":    "Payment Ratio",
        "Your Value":     f"{pay_ratio:.2f}x",
        "Risk Signal":    "🟢 Good" if pay_ratio > 0.5 else "🟡 Low" if pay_ratio > 0.2 else "🔴 Very Low",
        "What It Means":  "Ratio of avg payment made vs avg bill amount"
    },
    {
        "Risk Factor":    "Max Payment Delay",
        "Your Value":     f"{max_delay} months",
        "Risk Signal":    "🔴 High Risk" if max_delay >= 2 else "🟡 Watch" if max_delay == 1 else "🟢 Clean",
        "What It Means":  "Worst single payment delay in last 6 months"
    },
    {
        "Risk Factor":    "Delay Count",
        "Your Value":     f"{delay_count} / 6 months",
        "Risk Signal":    "🔴 Frequent" if delay_count >= 3 else "🟡 Occasional" if delay_count >= 1 else "🟢 None",
        "What It Means":  "Number of months with any payment delay"
    },
    {
        "Risk Factor":    "Credit Limit",
        "Your Value":     f"NT$ {limit_bal:,}",
        "Risk Signal":    "🟢 High" if limit_bal > 200000 else "🟡 Medium" if limit_bal > 50000 else "🔴 Low",
        "What It Means":  "Higher limits typically indicate trusted borrowers"
    },
    {
        "Risk Factor":    "Bill Trend",
        "Your Value":     f"NT$ {bill_trend:,}",
        "Risk Signal":    "🔴 Growing Debt" if bill_trend > 10000 else "🟡 Stable" if bill_trend > 0 else "🟢 Shrinking",
        "What It Means":  "Is outstanding debt growing or shrinking over time?"
    },
])

st.dataframe(
    factors_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Risk Factor":   st.column_config.TextColumn(width="medium"),
        "Your Value":    st.column_config.TextColumn(width="small"),
        "Risk Signal":   st.column_config.TextColumn(width="small"),
        "What It Means": st.column_config.TextColumn(width="large"),
    }
)

st.markdown("---")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:grey; font-size:13px; padding-top:10px'>
    Built by <b>Abhinav Srivastav</b> | AMPBA, ISB Hyderabad | 
    <a href='https://github.com/Abhinav19-isb/credit-risk-scorecard' target='_blank'>GitHub Repo</a> |
    Trained on UCI Credit Card Default Dataset (CC BY 4.0)
    </div>
    """,
    unsafe_allow_html=True
)
