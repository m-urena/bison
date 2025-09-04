import httpx, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from functools import lru_cache

st.set_page_config(page_title="Fund Dashboard", layout="wide")

FASTTRACK_BASE = "https://fasttrackapi.com/v1"

FASTTRACK_ACCOUNT  = "702528"
FASTTRACK_PASSWORD = "2FE9C5FA"
FASTTRACK_APPID    = "4967E757-E918-4253-B798-0EA79C654885"

@st.cache_data(ttl=3600)
def get_fasttrack_token():
    url = f"{FASTTRACK_BASE}/login"
    payload = {
        "username": FASTTRACK_ACCOUNT,
        "password": FASTTRACK_PASSWORD,
        "appid": FASTTRACK_APPID
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("Failed to retrieve FastTrack token")
    return token

def get_fasttrack_headers():
    return {"Authorization": f"Bearer {get_fasttrack_token()}"}

# ============================
# Fund Mapping
# ============================
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

# ============================
# Helper Functions
# ============================
def period_window(key):
    today = pd.Timestamp.today().normalize()
    if key == "YTD":
        start = pd.Timestamp(datetime(today.year,1,1))
    elif key == "1Y":
        start = today - pd.DateOffset(years=1)
    elif key == "3Y":
        start = today - pd.DateOffset(years=3)
    elif key == "5Y":
        start = today - pd.DateOffset(years=5)
    else:
        start = today - pd.DateOffset(years=1)
    return start, today

@st.cache_data(ttl=3600)
def fasttrack_total_return_index(ticker, start, end):
    url = f"{FASTTRACK_BASE}/series/totalreturn"
    params = {"symbols": ticker, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, headers=get_fasttrack_headers(), params=params)
        r.raise_for_status()
        data = r.json()
    if not data or "series" not in data or ticker not in data["series"]:
        return None
    df = pd.DataFrame(data["series"][ticker])
    if df.empty or "date" not in df or "value" not in df:
        return None
    s = pd.Series(df["value"].astype(float).values, index=pd.to_datetime(df["date"]))
    s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if s.empty:
        return None
    return s

def series_period_return_from_index(index_series, start, end):
    s = pd.Series(index_series).dropna().astype(float)
    s = s[(s.index >= start) & (s.index <= end)]
    if s.empty:
        return np.nan, pd.Series(dtype=float)
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return np.nan, pd.Series(dtype=float)
    tri_norm = s / base
    r = tri_norm.pct_change().dropna()
    total_ret = float(tri_norm.iloc[-1] - 1.0)
    return total_ret, r

def sharpe_ratio(series, rf_daily):
    z = pd.concat([pd.Series(series).dropna(), pd.Series(rf_daily).dropna()], axis=1).dropna()
    if z.empty:
        return np.nan
    ex = z.iloc[:,0] - z.iloc[:,1]
    vol = ex.std() * np.sqrt(252)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return float((ex.mean() * 252) / vol)

def sortino_ratio(series, rf_daily):
    z = pd.concat([pd.Series(series).dropna(), pd.Series(rf_daily).dropna()], axis=1).dropna()
    if z.empty:
        return np.nan
    ex = z.iloc[:,0] - z.iloc[:,1]
    dn = ex[ex < 0].std() * np.sqrt(252)
    if dn == 0 or pd.isna(dn):
        return np.nan
    return float((ex.mean() * 252) / dn)

def max_drawdown(ret_series):
    s = pd.Series(ret_series).dropna().astype(float)
    if s.empty:
        return np.nan
    w = (1 + s).cumprod()
    dd = 1 - w.div(w.cummax())
    return float(dd.max())

@st.cache_data(ttl=3600)
def fasttrack_metadata(ticker):
    url = f"{FASTTRACK_BASE}/securities/metadata"
    params = {"symbols": ticker}
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, headers=get_fasttrack_headers(), params=params)
        r.raise_for_status()
        data = r.json()
    return data.get("metadata", {}).get(ticker, {})

@lru_cache(maxsize=None)
def get_expense_ratio(ticker):
    meta = fasttrack_metadata(ticker)
    er = meta.get("expenseRatio")
    if er is None:
        return np.nan
    er = float(er)
    if er >= 5:
        er = er / 100.0
    if er > 0.5:
        er = er / 100.0
    return float(er)

def get_dividend_yield(ticker):
    meta = fasttrack_metadata(ticker)
    y = meta.get("dividendYield")
    if y is None:
        return np.nan
    return float(y) * 100.0

@st.cache_data(ttl=1800)
def rf_daily_series(start, end):
    idx = pd.date_range(start=start, end=end, freq="B")
    return pd.Series(0.0, index=idx, name="RF")

def build_pair_series(ticker, start, end):
    tri = fasttrack_total_return_index(ticker, start, end)
    if tri is None or tri.empty:
        return np.nan, pd.Series(dtype=float)
    total, rets = series_period_return_from_index(tri, start, end)
    return total, rets

@st.cache_data(ttl=1800)
def build_vs_benchmark(period_key):
    start, end = period_window(period_key)
    rf = rf_daily_series(start, end)
    rows = []
    for fund, meta in fund_map.items():
        bench = meta["benchmark"]
        f_tot, f_ret = build_pair_series(fund, start, end)
        b_tot, b_ret = build_pair_series(bench, start, end)
        if pd.isna(f_tot) or pd.isna(b_tot) or f_ret.empty or b_ret.empty:
            continue
        idx = f_ret.index.intersection(b_ret.index).intersection(rf.index)
        if len(idx) < 60:
            continue
        fr = f_ret.reindex(idx)
        br = b_ret.reindex(idx)
        rfr = rf.reindex(idx).fillna(0)
        rows.append({
            "Fund": fund,
            "Benchmark": bench,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            f"Fund Total Return ({period_key})": f_tot,
            f"Benchmark Total Return ({period_key})": b_tot,
            f"Excess Total Return ({period_key})": f_tot - b_tot,
            "Sharpe": sharpe_ratio(fr, rfr),
            "Sortino": sortino_ratio(fr, rfr),
            "Max Drawdown": max_drawdown(fr),
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def build_vs_each_other(period_key):
    start, end = period_window(period_key)
    rf = rf_daily_series(start, end)
    rows = []
    for fund, meta in fund_map.items():
        tot, r = build_pair_series(fund, start, end)
        if pd.isna(tot) or r.empty:
            continue
        idx = r.index.intersection(rf.index)
        if len(idx) < 60:
            continue
        rr = r.reindex(idx)
        rows.append({
            "Fund": fund,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            f"Total Return ({period_key})": tot,
            "Sharpe": sharpe_ratio(rr, rf.reindex(idx).fillna(0)),
            "Sortino": sortino_ratio(rr, rf.reindex(idx).fillna(0)),
            "Max Drawdown": max_drawdown(rr),
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    return pd.DataFrame(rows)

def build_custom_comparison(tickers, period_key):
    start, end = period_window(period_key)
    rf = rf_daily_series(start, end)
    rows = []
    rets_map = {}
    for t in tickers:
        tot, r = build_pair_series(t, start, end)
        if pd.isna(tot) or r.empty:
            continue
        idx = r.index.intersection(rf.index)
        if len(idx) < 60:
            continue
        rr = r.reindex(idx)
        rows.append({
            "Fund": t,
            f"Total Return ({period_key})": tot,
            "Sharpe": sharpe_ratio(rr, rf.reindex(idx).fillna(0)),
            "Sortino": sortino_ratio(rr, rf.reindex(idx).fillna(0)),
            "Max Drawdown": max_drawdown(rr),
            "Expense Ratio": get_expense_ratio(t),
            "Dividend Yield %": get_dividend_yield(t)
        })
        rets_map[t] = rr
    df = pd.DataFrame(rows)
    corr_df = None
    if len(rets_map) >= 2:
        rets_df = pd.DataFrame(rets_map)
        corr_df = rets_df.corr()
    return df, corr_df
