import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="ETF Traffic Lights", layout="wide")

# ----------------------------
# ETF MAP
# ----------------------------
etf_map = {
    "IBIT":  {"benchmark": "IBIT", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Crypto"},
    "IQDY":  {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Income",       "strategy": "Foreign"},
    "QQQ":   {"benchmark": "QQQ",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "DNLIX": {"benchmark": "SPY",  "asset_class": "Alt", "purpose": "Preservation", "strategy": "Hedged"},
    "AVUV":  {"benchmark": "IJR",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Value"},
    "GRID":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "XMMO":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "PAVE":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "OVF":   {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Foreign"},
    "SCHD":  {"benchmark": "IWD",  "asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"},
    "OVLH":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "DGRW":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"},
    "FLQM":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Balanced"},
    "KHPI":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "IEF":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "ICSH":  {"benchmark": "BIL","asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "CGSM":  {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Municipal"},
    "SHYD":  {"benchmark": "HYD",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "BIL":   {"benchmark": "BIL","asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "ESIIX": {"benchmark": "HYG",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "SHY":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "OVB":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Overlay"},
    "OVT":   {"benchmark": "VCSH", "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Overlay"},
    "CLOB":  {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "HYMB":  {"benchmark": "HYD",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "MBSF":  {"benchmark": "MBB",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "IAU":   {"benchmark": "GLD",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IGLD":  {"benchmark": "GLD",  "asset_class": "Alts", "purpose": "Preservation", "strategy": "Commodity"},
    "IEI":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "NAGRX": {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Core Bond"},
    "IWF":   {"benchmark": "IWF",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "OVS":   {"benchmark": "IJR",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Overlay"},
    "OVL":   {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Overlay"},
    "OVM":   {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Municipal"},
    "CLOI":  {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "FIW":   {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "PEY":   {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Income", "strategy": "Dividend"},
    "GSIMX": {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Foreign"},
    "DFNDX": {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "PSFF":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Income", "strategy": "Hedged"},
    "CPITX": {"benchmark": "HYG",  "asset_class": "Fixed Income", "purpose": "Income", "strategy": "High Yield"}
}

# ----------------------------
# HELPERS
# ----------------------------
def sortino_ratio(series, rf_series_daily):
    z = pd.concat([pd.Series(series).dropna(), rf_series_daily], axis=1).dropna()
    if z.empty: return np.nan
    ex = z.iloc[:,0] - z.iloc[:,1]
    dn = ex[ex < 0].std() * np.sqrt(252)
    if dn == 0 or np.isnan(dn): return np.nan
    return (ex.mean() * 252) / dn

def max_drawdown(r):
    s = pd.Series(r).dropna().astype(float)
    if s.empty: return np.nan
    w = (1 + s).cumprod()
    return float((1 - w.div(w.cummax())).max())

def get_expense_ratio(ticker): return np.nan
def get_dividend_yield(ticker): return np.nan

# ----------------------------
# CALCULATIONS
# ----------------------------
def build_vs_benchmark(rets):
    rows = []
    for etf, meta in etf_map.items():
        bench = meta["benchmark"]
        if etf not in rets.columns or bench not in rets.columns: continue
        z = rets.loc[:, [etf, bench]].dropna()
        if z.shape[0] < 60: continue
        etf_ret, bench_ret = z.iloc[:,0], z.iloc[:,1]
        rows.append({
            "ETF": etf,
            "Benchmark": bench,
            "Purpose": meta["purpose"],
            "Asset Class": meta["asset_class"],
            "Strategy": meta["strategy"],
            "ETF Return (annualized)": etf_ret.mean()*252,
            "Benchmark Return (annualized)": bench_ret.mean()*252,
            "Excess Return (annualized)": (etf_ret - bench_ret).mean()*252,
            "Excess Sortino": sortino_ratio(etf_ret, pd.Series(0, index=etf_ret.index)) - sortino_ratio(bench_ret, pd.Series(0, index=bench_ret.index)),
            "Excess Max Drawdown": max_drawdown(etf_ret) - max_drawdown(bench_ret),
            "Expense Ratio": get_expense_ratio(etf),
            "Dividend Yield %": get_dividend_yield(etf)
        })
    df = pd.DataFrame(rows)
    df["Points"] = ((df["Excess Return (annualized)"] > -0.01).astype(int) + (df["Excess Sortino"] > -0.05).astype(int) + (df["Dividend Yield %"].fillna(0)/100 > 0.025).astype(int))
    df["Color"] = np.select([df["Points"]>=2, df["Points"]==1], ["Green","Yellow"], default="Red")
    return df

def build_vs_each_other(rets):
    rows = []
    for etf, meta in etf_map.items():
        if etf not in rets.columns: continue
        r = rets.loc[:, [etf]].dropna().iloc[:,0]
        if r.shape[0] < 60: continue
        rows.append({
            "ETF": etf,
            "Purpose": meta["purpose"],
            "Asset Class": meta["asset_class"],
            "Strategy": meta["strategy"],
            "Return (annualized)": r.mean()*252,
            "Sortino": sortino_ratio(r, pd.Series(0, index=r.index)),
            "Max Drawdown": max_drawdown(r),
            "Expense Ratio": get_expense_ratio(etf),
            "Dividend Yield %": get_dividend_yield(etf)
        })
    df = pd.DataFrame(rows)
    def quartile_points(s):
        r = pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")
        return pd.cut(r, bins=[0,0.25,0.5,0.75,1.0000001], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)
    df["Points"] = quartile_points(df["Return (annualized)"]) + quartile_points(df["Sortino"]) + quartile_points(df["Dividend Yield %"])
    df["Color"] = np.select([df["Points"]<=2, df["Points"]<=6], ["Red","Yellow"], default="Green")
    return df

def style_table(df, view):
    order_map = {"Green":0, "Yellow":1, "Red":2}
    df = df.assign(_c=df["Color"].map(order_map)).sort_values(["_c","Points"], ascending=[True,False]).drop(columns=["_c"])
    def color_css(v):
        if v=="Green": return "background-color:#d6f5d6; color:#0a0a0a"
        if v=="Yellow": return "background-color:#fff5bf; color:#0a0a0a"
        if v=="Red": return "background-color:coral; color:#0a0a0a"
        return ""
    styler = df.style.map(color_css, subset=["Color"])
    try: styler = styler.hide(axis="index")
    except: pass
    return styler

# ----------------------------
# STREAMLIT APP
# ----------------------------
st.sidebar.title("ETF Traffic Lights")
mode = st.sidebar.selectbox("View", ["Vs Benchmark","Vs Each Other"])
start_date = st.sidebar.date_input("Start Date", value=date(2020,1,1))

# fake price data for now
px = yf.download(list(etf_map.keys()), start=start_date, end=date.today(), progress=False)["Close"]
rets = px.pct_change().dropna()

df = build_vs_benchmark(rets) if mode=="Vs Benchmark" else build_vs_each_other(rets)

# sidebar filters
purpose_filter = st.sidebar.multiselect("Filter by Purpose", sorted(df["Purpose"].dropna().unique()), default=sorted(df["Purpose"].dropna().unique()))
asset_filter   = st.sidebar.multiselect("Filter by Asset Class", sorted(df["Asset Class"].dropna().unique()), default=sorted(df["Asset Class"].dropna().unique()))
ticker_search  = st.sidebar.text_input("Search Ticker (e.g. SCHD)").upper()

df = df[df["Purpose"].isin(purpose_filter) & df["Asset Class"].isin(asset_filter)]
if ticker_search:
    df = df[df["ETF"].str.contains(ticker_search)]

st.subheader(mode)
st.dataframe(style_table(df, mode), use_container_width=True)
