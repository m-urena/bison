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
    "CPITX": {"benchmark": "HYG", "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"},
}

def style_table(df):
    if df.empty:
        return df

    def pct_fmt(v):
        try:
            return f"{float(v)*100:.2f}%"
        except:
            return v if isinstance(v, str) else ""

    def ratio_fmt(v):
        try:
            return f"{float(v)*100:.2f}%"
        except:
            return v if isinstance(v, str) else ""

    def num_fmt(v):
        try:
            return f"{float(v):.2f}"
        except:
            return v if isinstance(v, str) else ""

    fmt = {}
    for col in df.columns:
        if "Return" in col or "Yield" in col or "Drawdown" in col:
            fmt[col] = pct_fmt
        elif "Expense Ratio" in col:
            fmt[col] = ratio_fmt
        elif "Sharpe" in col or "Sortino" in col:
            fmt[col] = num_fmt

    styler = df.style.format(fmt)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler


def get_metric_columns(period_key):
    if period_key in ["YTD", "1 Year"]:
        return "Sharpe 1Y", "Sortino 1Y", "Max Drawdown 1Y"
    elif "3" in period_key:
        return "Sharpe 3Y", "Sortino 3Y", "Max Drawdown 3Y"
    elif "5" in period_key:
        return "Sharpe 5Y", "Sortino 5Y", "Max Drawdown 5Y"
    else:
        return None, None, None

st.sidebar.title("Fund Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
period_key = st.sidebar.selectbox("Period", ["YTD", "1 Year", "3 Year Total", "3 Year Annualized", "5 Year Total", "5 Year Annualized"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark", "Vs Each Other"], index=0)

if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    raw = raw.rename(columns={raw.columns[0]: "Ticker", raw.columns[1]: "Name"})

    sharpe_col, sortino_col, md_col = get_metric_columns(period_key)

    rows = []

    if mode == "Vs Benchmark":
        for fund, meta in fund_map.items():
            if fund not in raw["Ticker"].values:
                continue
            row = raw.loc[raw["Ticker"] == fund].iloc[0]

            bench = meta["benchmark"]
            if bench not in raw["Ticker"].values:
                continue
            bench_row = raw.loc[raw["Ticker"] == bench].iloc[0]

            fund_val = row.get(period_key, None)
            bench_val = bench_row.get(period_key, None)

            if pd.isna(fund_val) or pd.isna(bench_val):
                continue

            rows.append({
                "Fund": fund,
                "Benchmark": bench,
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Fund Return ({period_key})": fund_val,
                f"Benchmark Return ({period_key})": bench_val,
                f"Excess Return ({period_key})": fund_val - bench_val,
                "Sharpe": row.get(sharpe_col, "No Data"),
                "Sortino": row.get(sortino_col, "No Data"),
                "Max Drawdown": row.get(md_col, "No Data"),
                "Expense Ratio": row.get("Expense Ratio", None),
                "Dividend Yield %": row.get("Yield", None)
            })

    elif mode == "Vs Each Other":
        for fund, meta in fund_map.items():
            if fund not in raw["Ticker"].values:
                continue
            row = raw.loc[raw["Ticker"] == fund].iloc[0]

            fund_val = row.get(period_key, None)
            if pd.isna(fund_val):
                continue

            rows.append({
                "Fund": fund,
                "Asset Class": meta["asset_class"],
                "Purpose": meta["purpose"],
                "Strategy": meta["strategy"],
                f"Return ({period_key})": fund_val,
                "Sharpe": row.get(sharpe_col, "No Data"),
                "Sortino": row.get(sortino_col, "No Data"),
                "Max Drawdown": row.get(md_col, "No Data"),
                "Expense Ratio": row.get("Expense Ratio", None),
                "Dividend Yield %": row.get("Yield", None)
            })

    df = pd.DataFrame(rows)

    st.subheader(mode + f" — {period_key}")
    if df.empty:
        st.info("No rows for current selection.")
    else:
         st.dataframe(style_table(df), use_container_width=True)

    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()
