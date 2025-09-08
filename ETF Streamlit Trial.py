import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fund Dashboard", layout="wide")

# ============================================
# Config
# ============================================
scoring_weights = {
    "Excess Return": 0.35,
    "Sharpe": 0.10,
    "Sortino": 0.20,
    "Max Drawdown": 0.10,
    "Expense Ratio": 0.15,
    "Dividend Yield %": 0.10,
}

fund_map = {
    "IBIT":  {"benchmark": "IBIT", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "IQDY":  {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Income",       "strategy": "Foreign"},
    "QQQ":   {"benchmark": "QQQ",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "DNLIX": {"benchmark": "SPY",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Hedged"},
    "AVUV":  {"benchmark": "IJR",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Small Cap"},
    "GRID":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "XMMO":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "PAVE":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "OVF":   {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Foreign"},
    "SCHD":  {"benchmark": "IWD",  "asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"},
    "OVLH":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "DGRW":  {"benchmark": "SCHD","asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"},
    "FLQM":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Mid Cap"},
    "KHPI":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "IEF":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "ICSH":  {"benchmark": "BIL",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "CGSM":  {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Municipal"},
    "SHYD":  {"benchmark": "HYD",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "BIL":   {"benchmark": "BIL",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "ESIIX": {"benchmark": "HYG",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "SHY":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "OVB":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Core Bond"},
    "OVT":   {"benchmark": "VCSH", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Short Term Bond"},
    "CLOB":  {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "HYMB":  {"benchmark": "HYD",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "MBSF":  {"benchmark": "MBB",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "IAU":   {"benchmark": "GLD",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IGLD":  {"benchmark": "GLD",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IEI":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "NAGRX": {"benchmark": "AGG",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Core Bond"},
    "IWF":   {"benchmark": "IWF",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "OVS":   {"benchmark": "IJR",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Small Cap"},
    "OVL":   {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Large Cap"},
    "OVM":   {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Municipal"},
    "CLOI":  {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "FIW":   {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "PEY":   {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Income", "strategy": "Dividend"},
    "GSIMX": {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Foreign"},
    "DFNDX": {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "PSFF":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Income", "strategy": "Hedged"},
    "CPITX": {"benchmark": "HYG",  "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"}
}

# ============================================
# Helpers
# ============================================
def normalize(series, reverse=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.nunique() <= 1:
        return pd.Series(0, index=s.index)
    if reverse:
        s = -s
    return (s - s.min()) / (s.max() - s.min())

def safe_col(df, col):
    return df[col] if col in df.columns else pd.Series([np.nan]*len(df), index=df.index)

def add_scores(df, period_key, mode):
    df = df.copy()

    if period_key == "YTD":
        ex_col, sh_col, so_col, md_col = "YTD", "Sharpe 1Y", "Sortino 1Y", "Max Drawdown 1Y"
    elif period_key == "1Y":
        ex_col, sh_col, so_col, md_col = "1 Year", "Sharpe 1Y", "Sortino 1Y", "Max Drawdown 1Y"
    elif period_key == "3Y":
        ex_col, sh_col, so_col, md_col = "3 Year Annualized", "Sharpe 3Y", "Sortino 3Y", "Max Drawdown 3Y"
    else:  # 5Y
        ex_col, sh_col, so_col, md_col = "5 Year Annualized", "Sharpe 5Y", "Sortino 5Y", "Max Drawdown 5Y"

    df["_ex"] = normalize(safe_col(df, ex_col))
    df["_sh"] = normalize(safe_col(df, sh_col))
    df["_so"] = normalize(safe_col(df, so_col))
    df["_md"] = normalize(safe_col(df, md_col), reverse=True)
    df["_er"] = normalize(safe_col(df, "Expense Ratio"), reverse=True)
    df["_dy"] = normalize(safe_col(df, "Yield"))

    df["Score"] = (
        df["_ex"] * scoring_weights["Excess Return"]
        + df["_sh"] * scoring_weights["Sharpe"]
        + df["_so"] * scoring_weights["Sortino"]
        + df["_md"] * scoring_weights["Max Drawdown"]
        + df["_er"] * scoring_weights["Expense Ratio"]
        + df["_dy"] * scoring_weights["Dividend Yield %"]
    )

    return df.drop(columns=["_ex", "_sh", "_so", "_md", "_er", "_dy"])

def style_table(df):
    if df.empty:
        return df
    fmt = {}
    pct_cols = [c for c in df.columns if any(k in c for k in ["Return", "Drawdown", "Ratio", "Yield"])]
    for c in pct_cols:
        fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%" if abs(float(v)) < 1 else f"{float(v):.2f}"
    if "Score" in df.columns:
        fmt["Score"] = lambda v: "" if pd.isna(v) else f"{float(v):.3f}"
    styler = df.style.format(fmt)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

# ============================================
# Streamlit App
# ============================================
st.sidebar.title("Fund Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
period_key = st.sidebar.selectbox("Period", ["YTD", "1Y", "3Y", "5Y"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark", "Vs Each Other"], index=0)

if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    raw.columns = raw.columns.str.strip()

    purpose_options = ["All"] + sorted(raw["Purpose"].dropna().unique().tolist()) if "Purpose" in raw.columns else ["All"]
    selected_purpose = st.sidebar.selectbox("Filter by Purpose", purpose_options)

    df = raw.copy()
    if selected_purpose != "All" and "Purpose" in df.columns:
        df = df[df["Purpose"] == selected_purpose]

    df = add_scores(df, period_key, mode)

    st.subheader(mode + f" — {period_key}")
    if df.empty:
        st.info("No rows for current selection.")
    else:
        st.dataframe(style_table(df.sort_values("Score", ascending=False)), use_container_width=True)

    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()
else:
    st.info("Please upload an Excel file to continue.")
