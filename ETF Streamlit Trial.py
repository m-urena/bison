import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fund Dashboard", layout="wide")

# ---------------- SAFE NUMBER CLEANER ----------------
def safe_number(x):
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("%"):
                return float(x.replace("%", "")) / 100.0
            if x.lower() == "no data":
                return np.nan
        return float(x)
    except:
        return np.nan

# ---------------- NORMALIZATION ----------------
def normalize(series, reverse=False):
    s = pd.Series(series, dtype=float).copy()
    if reverse:
        s = -s
    if s.nunique() <= 1:
        return pd.Series([1] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# ---------------- WEIGHTS ----------------
scoring_weights = {
    "Excess Return": 0.35,
    "Sharpe": 0.10,
    "Sortino": 0.20,
    "Max Drawdown": 0.10,
    "Expense Ratio": 0.15,
    "Dividend Yield %": 0.10
}

# ---------------- PERIOD MAPS ----------------
period_map = {
    "YTD": "YTD",
    "1Y": "1 Year",
    "3Y": "3 Year Annualized",
    "5Y": "5 Year Annualized"
}

sharpe_map = {
    "YTD": "Sharpe 1Y",
    "1Y": "Sharpe 1Y",
    "3Y": "Sharpe 3Y",
    "5Y": "Sharpe 5Y"
}

sortino_map = {
    "YTD": "Sortino 1Y",
    "1Y": "Sortino 1Y",
    "3Y": "Sortino 3Y",
    "5Y": "Sortino 5Y"
}

md_map = {
    "YTD": "Max Drawdown 1Y",
    "1Y": "Max Drawdown 1Y",
    "3Y": "Max Drawdown 3Y",
    "5Y": "Max Drawdown 5Y"
}

# ---------------- FUND MAP ----------------
fund_map = {
    "IBIT": {"benchmark": "IBIT", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "IQDY": {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Income", "strategy": "Foreign"},
    "QQQ": {"benchmark": "QQQ", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "DNLIX": {"benchmark": "SPY", "asset_class": "Alts", "purpose": "Preservation", "strategy": "Hedged"},
    "AVUV": {"benchmark": "IJR", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Small Cap"},
    "GRID": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "XMMO": {"benchmark": "IJH", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "PAVE": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "OVF": {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Foreign"},
    "SCHD": {"benchmark": "IWD", "asset_class": "Equity", "purpose": "Income", "strategy": "Dividend"},
    "OVLH": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "DGRW": {"benchmark": "SCHD", "asset_class": "Equity", "purpose": "Income", "strategy": "Dividend"},
    "FLQM": {"benchmark": "IJH", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Mid Cap"},
    "KHPI": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "IEF": {"benchmark": "AGG", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "ICSH": {"benchmark": "BIL", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "CGSM": {"benchmark": "MUB", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "Municipal"},
    "SHYD": {"benchmark": "HYD", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"},
    "BIL": {"benchmark": "BIL", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "ESIIX": {"benchmark": "HYG", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"},
    "SHY": {"benchmark": "AGG", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "OVB": {"benchmark": "AGG", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Core Bond"},
    "OVT": {"benchmark": "VCSH", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Short Term Bond"},
    "CLOB": {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "Alt Credit"},
    "HYMB": {"benchmark": "HYD", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"},
    "MBSF": {"benchmark": "MBB", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "Alt Credit"},
    "IAU": {"benchmark": "GLD", "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IGLD": {"benchmark": "GLD", "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IEI": {"benchmark": "AGG", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "NAGRX": {"benchmark": "AGG", "asset_class": "Alts", "purpose": "Preservation", "strategy": "Core Bond"},
    "IWF": {"benchmark": "IWF", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "OVS": {"benchmark": "IJR", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Small Cap"},
    "OVL": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Large Cap"},
    "OVM": {"benchmark": "MUB", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Municipal"},
    "CLOI": {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "Alt Credit"},
    "FIW": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "PEY": {"benchmark": "IJH", "asset_class": "Equity", "purpose": "Income", "strategy": "Dividend"},
    "GSIMX": {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Foreign"},
    "DFNDX": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "PSFF": {"benchmark": "SPY", "asset_class": "Equity", "purpose": "Income", "strategy": "Hedged"},
    "CPITX": {"benchmark": "HYG", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"}
}

def add_scores(df, period_key, mode):
    df = df.copy()

    if period_key in ["YTD", "1Y"]:
        ex_col = "Excess Return (1Y)" if mode == "Vs Benchmark" else "Return (1Y)"
        sh_col = "Sharpe 1Y"
        so_col = "Sortino 1Y"
        md_col = "Max Drawdown 1Y"
    elif period_key == "3Y":
        ex_col = "Excess Return (3Y)" if mode == "Vs Benchmark" else "Return (3Y)"
        sh_col = "Sharpe 3Y"
        so_col = "Sortino 3Y"
        md_col = "Max Drawdown 3Y"
    else:  # 5Y
        ex_col = "Excess Return (5Y)" if mode == "Vs Benchmark" else "Return (5Y)"
        sh_col = "Sharpe 5Y"
        so_col = "Sortino 5Y"
        md_col = "Max Drawdown 5Y"

    df["_ex"] = normalize(df[ex_col])
    df["_sh"] = normalize(df[sh_col])
    df["_so"] = normalize(df[so_col])
    df["_md"] = normalize(df[md_col], reverse=True)
    df["_er"] = normalize(df["Expense Ratio"], reverse=True)
    df["_dy"] = normalize(df["Dividend Yield %"])

    df["Score"] = (
        df["_ex"] * scoring_weights["Excess Return"]
        + df["_sh"] * scoring_weights["Sharpe"]
        + df["_so"] * scoring_weights["Sortino"]
        + df["_md"] * scoring_weights["Max Drawdown"]
        + df["_er"] * scoring_weights["Expense Ratio"]
        + df["_dy"] * scoring_weights["Dividend Yield %"]
    )

    return df.drop(columns=["_ex", "_sh", "_so", "_md", "_er", "_dy"])


# ---------------- UI ----------------
st.sidebar.title("Fund Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
period_key = st.sidebar.selectbox("Period", ["YTD", "1Y", "3Y", "5Y"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark", "Vs Each Other"], index=0)

if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    raw = raw.rename(columns={raw.columns[0]: "Ticker", raw.columns[1]: "Name"})

    rows = []
    for fund, meta in fund_map.items():
        fund_row = raw.loc[raw["Ticker"] == fund]
        bench_row = raw.loc[raw["Ticker"] == meta["benchmark"]]

        if fund_row.empty:
            continue
        fund_row = fund_row.iloc[0]
        bench_row = bench_row.iloc[0] if not bench_row.empty else None

        col = period_map[period_key]
        f_ret = safe_number(fund_row[col])
        b_ret = safe_number(bench_row[col]) if bench_row is not None else np.nan

        sharpe_col = sharpe_map[period_key]
        sortino_col = sortino_map[period_key]
        md_col = md_map[period_key]

        if mode == "Vs Benchmark":
            rows.append({
                "Fund": fund,
                "Benchmark": meta["benchmark"],
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Fund Return ({period_key})": f_ret,
                f"Benchmark Return ({period_key})": b_ret,
                f"Excess Return ({period_key})": f_ret - b_ret if pd.notna(f_ret) and pd.notna(b_ret) else np.nan,
                "Sharpe " + period_key: safe_number(fund_row.get(sharpe_col, np.nan)),
                "Sortino " + period_key: safe_number(fund_row.get(sortino_col, np.nan)),
                "Max Drawdown " + period_key: safe_number(fund_row.get(md_col, np.nan)),
                "Expense Ratio": safe_number(fund_row["Expense Ratio"]),
                "Dividend Yield %": safe_number(fund_row["Yield"])
            })
        else:
            rows.append({
                "Fund": fund,
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Return ({period_key})": f_ret,
                "Sharpe " + period_key: safe_number(fund_row.get(sharpe_col, np.nan)),
                "Sortino " + period_key: safe_number(fund_row.get(sortino_col, np.nan)),
                "Max Drawdown " + period_key: safe_number(fund_row.get(md_col, np.nan)),
                "Expense Ratio": safe_number(fund_row["Expense Ratio"]),
                "Dividend Yield %": safe_number(fund_row["Yield"])
            })

    df = pd.DataFrame(rows)

    if not df.empty:
       df = add_scores(df, period_key, mode)
        df = df.sort_values("Score", ascending=False)

        purpose_options = ["All"] + sorted(df["Purpose"].dropna().unique().tolist())
        selected_purpose = st.sidebar.selectbox("Filter by Purpose", purpose_options)

        if selected_purpose != "All":
            df = df[df["Purpose"] == selected_purpose]

        st.subheader(mode + f" — {period_key}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No rows for current selection.")
