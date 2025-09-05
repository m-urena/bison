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

def style_table(df):
    if df.empty:
        return df
    def fmt_percent(v):
        if pd.isna(v) or v == "No Data":
            return ""
        try:
            v = float(str(v).replace("%", "").strip())
            if v < 1:
                v *= 100
            return f"{v:.2f}%"
        except Exception:
            return v
    def fmt_ratio(v):
        if pd.isna(v) or v == "No Data":
            return ""
        try:
            return f"{float(v):.3f}"
        except Exception:
            return v
    fmt = {}
    for c in df.columns:
        if "Return" in c or "Drawdown" in c or c in ["Dividend Yield %", "Expense Ratio"]:
            fmt[c] = fmt_percent
        if "Sharpe" in c or "Sortino" in c:
            fmt[c] = fmt_ratio
    styler = df.style.format(fmt)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

st.sidebar.title("Fund Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
period_key = st.sidebar.selectbox("Period", ["YTD","1 Year","3 Year Total","3 Year Annualized","5 Year Total","5 Year Annualized"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark","Vs Each Other"], index=0)

if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    raw = raw.rename(columns={raw.columns[0]: "Ticker", raw.columns[1]: "Name"})

    sharpe_col = f"Sharpe {period_key.split()[0]}"
    sortino_col = f"Sortino {period_key.split()[0]}"
    if "Year" in period_key:
        md_col = f"Max Drawdown {period_key.split()[0]}"
    else:
        md_col = f"Max Drawdown {period_key.split()[0]}"

    if mode == "Vs Benchmark":
        rows = []
        for _, row in raw.iterrows():
            fund = str(row["Ticker"]).replace("M:","")
            if fund not in fund_map:
                continue
            bench = fund_map[fund]["benchmark"]
            bench_row = raw[raw["Ticker"] == bench]
            if bench_row.empty:
                continue
            bench_row = bench_row.iloc[0]
            try:
                fund_val = float(str(row[period_key]).replace("%",""))
                bench_val = float(str(bench_row[period_key]).replace("%",""))
                excess = fund_val - bench_val
            except Exception:
                fund_val, bench_val, excess = np.nan, np.nan, np.nan
            rows.append({
                "Fund": fund,
                "Benchmark": bench,
                "Asset Class": fund_map[fund]["asset_class"],
                "Purpose": fund_map[fund]["purpose"],
                "Strategy": fund_map[fund]["strategy"],
                f"Fund Return ({period_key})": fund_val,
                f"Benchmark Return ({period_key})": bench_val,
                f"Excess Return ({period_key})": excess,
                "Sharpe": row.get(sharpe_col),
                "Sortino": row.get(sortino_col),
                "Max Drawdown": row.get(md_col),
                "Expense Ratio": row.get("Expense Ratio"),
                "Dividend Yield %": row.get("Yield")
            })
        df = pd.DataFrame(rows)
        st.subheader(f"Funds vs Benchmark — {period_key}")
        st.dataframe(style_table(df), use_container_width=True)

    elif mode == "Vs Each Other":
        rows = []
        for _, row in raw.iterrows():
            fund = str(row["Ticker"]).replace("M:","")
            meta = fund_map.get(fund,{})
            rows.append({
                "Fund": fund,
                "Asset Class": meta.get("asset_class",""),
                "Purpose": meta.get("purpose",""),
                "Strategy": meta.get("strategy",""),
                f"Total Return ({period_key})": row[period_key],
                "Sharpe": row.get(sharpe_col),
                "Sortino": row.get(sortino_col),
                "Max Drawdown": row.get(md_col),
                "Expense Ratio": row.get("Expense Ratio"),
                "Dividend Yield %": row.get("Yield")
            })
        df = pd.DataFrame(rows)
        st.subheader(f"Funds vs Each Other — {period_key}")
        st.dataframe(style_table(df), use_container_width=True)

    custom_tickers = st.sidebar.text_input("Enter tickers for comparison")
    custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    if custom_list:
        st.subheader("Custom Fund Comparison")
        custom_df = raw[raw["Ticker"].str.replace("M:","").isin(custom_list)].copy()
        if custom_df.empty:
            st.info("No valid data for entered tickers.")
        else:
            st.dataframe(style_table(custom_df), use_container_width=True)

    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()

else:
    st.info("Please upload your Excel file to get started.")
