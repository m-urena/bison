import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fund Dashboard", layout="wide")

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

@st.cache_data
def load_data(uploaded_file):
    raw = pd.read_excel(uploaded_file)
    raw.columns = raw.columns.str.strip()
    raw = raw.rename(columns={raw.columns[0]: "Ticker", raw.columns[1]: "Name"})
    return raw

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

def style_table(df):
    if df.empty:
        return df
    fmt = {}
    pct_cols = [c for c in df.columns if "Return" in c or "Drawdown" in c or c in ["Expense Ratio", "Dividend Yield %"]]
    for c in pct_cols:
        fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%"
    if "Sharpe" in df.columns:
        fmt["Sharpe"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    if "Sortino" in df.columns:
        fmt["Sortino"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    styler = df.style.format(fmt)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

st.sidebar.title("Fund Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
period_key = st.sidebar.selectbox("Period", ["YTD", "1 Year", "3 Year", "5 Year"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark", "Vs Each Other"], index=0)

if uploaded_file:
    raw = load_data(uploaded_file)

    suffix_map = {"YTD": "1Y", "1 Year": "1Y", "3 Year": "3Y", "5 Year": "5Y"}
    suffix = suffix_map.get(period_key, "1Y")
    sharpe_col = f"Sharpe {suffix}"
    sortino_col = f"Sortino {suffix}"
    md_col = f"Max Drawdown {suffix}"

    rows = []
    for fund, meta in fund_map.items():
        if fund not in raw["Ticker"].values:
            continue
        fund_row = raw.loc[raw["Ticker"] == fund].iloc[0]
        bench = meta["benchmark"]
        if bench not in raw["Ticker"].values:
            continue
        bench_row = raw.loc[raw["Ticker"] == bench].iloc[0]

        if mode == "Vs Benchmark":
            rows.append({
                "Fund": fund,
                "Benchmark": bench,
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Fund Return ({period_key})": safe_number(fund_row[period_key]),
                f"Benchmark Return ({period_key})": safe_number(bench_row[period_key]),
                f"Excess Return ({period_key})": (
                    safe_number(fund_row[period_key]) - safe_number(bench_row[period_key])
                    if pd.notna(safe_number(fund_row[period_key])) and pd.notna(safe_number(bench_row[period_key]))
                    else np.nan
                ),
                "Sharpe": safe_number(fund_row.get(sharpe_col, np.nan)),
                "Sortino": safe_number(fund_row.get(sortino_col, np.nan)),
                "Max Drawdown": safe_number(fund_row.get(md_col, np.nan)),
                "Expense Ratio": safe_number(fund_row["Expense Ratio"]),
                "Dividend Yield %": safe_number(fund_row["Yield"])
            })
        else:
            rows.append({
                "Fund": fund,
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Return ({period_key})": safe_number(fund_row[period_key]),
                "Sharpe": safe_number(fund_row.get(sharpe_col, np.nan)),
                "Sortino": safe_number(fund_row.get(sortino_col, np.nan)),
                "Max Drawdown": safe_number(fund_row.get(md_col, np.nan)),
                "Expense Ratio": safe_number(fund_row["Expense Ratio"]),
                "Dividend Yield %": safe_number(fund_row["Yield"])
            })


    df = pd.DataFrame(rows)

    purpose_options = ["All"] + sorted(df["Purpose"].dropna().unique().tolist())
    selected_purpose = st.sidebar.selectbox("Filter by Purpose", purpose_options)
    if selected_purpose != "All":
        df = df[df["Purpose"] == selected_purpose]

    st.subheader(mode + f" — {period_key}")
    if df.empty:
        st.info("No rows for current selection.")
    else:
        st.dataframe(style_table(df), use_container_width=True)

    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()
