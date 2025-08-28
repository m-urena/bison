import numpy as np, pandas as pd, yfinance as yf, streamlit as st
from datetime import date, datetime
from functools import lru_cache
try:
    import mstarpy as ms
except Exception:
    ms = None

st.set_page_config(page_title="Fund Dashboard", layout="wide")

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
def mstar_total_return_index(ticker, start, end):
    if ms is None:
        return None
    try:
        f = ms.Funds(ticker)
        data = f.nav(start, end)
        if not data or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        if "date" not in df or "totalReturn" not in df:
            return None
        s = pd.to_datetime(df["date"])
        tri = pd.Series(df["totalReturn"].astype(float).values, index=s).sort_index()
        tri.index = tri.index.tz_localize(None)
        tri = tri[~tri.index.duplicated(keep="last")]
        tri = tri.dropna()
        if tri.empty:
            return None
        return tri
    except Exception:
        return None

@st.cache_data(ttl=3600)
def yahoo_adj_close(tickers, start, end):
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex) and ("Adj Close" in px.columns.get_level_values(0)):
        px = px["Adj Close"]
    elif "Adj Close" in px.columns:
        px = px["Adj Close"]
    else:
        if isinstance(px, pd.DataFrame) and "Close" in px.columns:
            px = px["Close"]
        elif isinstance(px, pd.Series):
            px = px.to_frame()
        else:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    px.index = pd.to_datetime(px.index, utc=True, errors="coerce").tz_convert(None)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    return px

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

def tri_from_adj_close(p):
    s = pd.Series(p).dropna().astype(float)
    if s.empty:
        return None
    tri = (s / s.iloc[0]) * 100.0
    return tri

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

@lru_cache(maxsize=None)
def get_expense_ratio(ticker):
    if ticker in ["NAGRX","DNLIX"]:
        return 0.0199
    if ticker in ["DFNDX"]:
        return 0.0204
    try:
        from yahooquery import Ticker as YQT
        yq = YQT(ticker)
        v = None
        prof = yq.fund_profile
        if isinstance(prof, dict) and ticker in prof:
            fees = prof[ticker].get("feesExpensesInvestment") or prof[ticker].get("feesExpensesOperating") or {}
            for k in ("annualReportExpenseRatio", "netExpRatio", "grossExpRatio", "expenseRatio"):
                if fees.get(k) is not None:
                    v = fees[k]
                    break
        if v is None:
            sd = yq.summary_detail
            if isinstance(sd, dict) and ticker in sd:
                for k in ("annualReportExpenseRatio", "expenseRatio"):
                    if sd[ticker].get(k) is not None:
                        v = sd[ticker][k]
                        break
        if v is None:
            yi = yf.Ticker(ticker).info or {}
            v = yi.get("expenseRatio")
        if v is None:
            return np.nan
        v = float(v)
        if v >= 5:
            v = v / 100.0
        if v > 0.5:
            v = v / 100.0
        return float(v)
    except Exception:
        return np.nan

def get_dividend_yield(ticker):
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        y = info.get("yield") or info.get("trailingAnnualDividendYield") or info.get("dividendYield")
        if y is not None and pd.notna(y) and float(y) > 0:
            return float(y) * 100.0
        divs = t.dividends
        if divs is None or divs.empty:
            return np.nan
        if getattr(divs.index, "tz", None) is not None:
            divs.index = divs.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        last12 = divs[divs.index >= cutoff]
        if last12.empty:
            return np.nan
        price = None
        fi = getattr(t, "fast_info", None)
        if fi:
            price = fi.get("last_price") or fi.get("previous_close")
        if not price:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        if not price:
            price = info.get("regularMarketPrice") or info.get("previousClose")
        if not price or pd.isna(price) or float(price) == 0.0:
            return np.nan
        total = float(last12.sum())
        return (total / float(price)) * 100.0
    except Exception:
        return np.nan

@st.cache_data(ttl=1800)
def rf_daily_series(start, end):
    try:
        from fredapi import Fred
        key = st.secrets.get("FRED_API_KEY", None)
        if not key:
            raise RuntimeError("no key")
        fred = Fred(api_key=key)
        rf = fred.get_series("DGS1", start).astype(float)/100.0
        idx = pd.date_range(start=start, end=end, freq="B")
        rf_df = pd.DataFrame(rf, columns=["RF"]).reindex(idx).ffill()
        rf_daily = (1.0 + rf_df["RF"]).pow(1/252.0) - 1.0
        rf_daily.name = "RF"
        return rf_daily
    except Exception:
        idx = pd.date_range(start=start, end=end, freq="B")
        return pd.Series(0.0, index=idx, name="RF")

def build_pair_series(ticker, start, end):
    tri = mstar_total_return_index(ticker, start, end)
    if tri is None:
        px = yahoo_adj_close([ticker], start, end)
        if isinstance(px, pd.DataFrame) and ticker in px.columns:
            tri = tri_from_adj_close(px[ticker])
        elif isinstance(px, pd.Series):
            tri = tri_from_adj_close(px)
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
        ex_ret = (fr - br).mean() * 252 if period_key in ("3Y","5Y") else (fr - br).mean() * 252
        rows.append({
            "Fund": fund,
            "Benchmark": bench,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            f"Fund Total Return ({period_key})": f_tot,
            f"Benchmark Total Return ({period_key})": b_tot,
            f"Excess Total Return ({period_key})": f_tot - b_tot,
            "Excess Sortino": sortino_ratio(fr, rfr) - sortino_ratio(br, rfr),
            "Excess Max Drawdown": max_drawdown(br) - max_drawdown(fr),
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        pts = ((df[f"Excess Total Return ({period_key})"] > -0.01).astype(int)
               + (df["Excess Sortino"] > -0.05).astype(int)
               + ((df["Dividend Yield %"].fillna(0)/100.0) > 0.025).astype(int))
        df["Points"] = pts
        df["Color"] = np.select([df["Points"]>=2, df["Points"]==1], ["Green","Yellow"], default="Red")
    return df

@st.cache_data(ttl=1800)
def build_vs_each_other(period_key):
    start, end = period_window(period_key)
    rf = rf_daily_series(start, end)
    rows = []
    rets_map = {}
    for fund, meta in fund_map.items():
        tot, r = build_pair_series(fund, start, end)
        if pd.isna(tot) or r.empty:
            continue
        idx = r.index.intersection(rf.index)
        if len(idx) < 60:
            continue
        rr = r.reindex(idx)
        sr = sortino_ratio(rr, rf.reindex(idx).fillna(0))
        mdd = max_drawdown(rr)
        rows.append({
            "Fund": fund,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            f"Total Return ({period_key})": tot,
            "Sortino": sr,
            "Max Drawdown": mdd,
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        def qpts(s):
            r = pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")
            return pd.cut(r, bins=[0,0.25,0.5,0.75,1.0000001], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)
        df["Points"] = qpts(df[f"Total Return ({period_key})"]) + qpts(df["Sortino"]) + qpts(df["Dividend Yield %"])
        df["Color"] = np.select([df["Points"]<=2, df["Points"]<=6], ["Red","Yellow"], default="Green")
    return df

def style_table(df):
    if df.empty:
        return df
    order_map = {"Green":0, "Yellow":1, "Red":2}
    if "Color" in df.columns and "Points" in df.columns:
        df = df.assign(_c=df["Color"].map(order_map)).sort_values(["_c","Points"], ascending=[True, False]).drop(columns=["_c"])
    def color_css(v):
        if v == "Green": return "background-color:#d6f5d6; color:#0a0a0a"
        if v == "Yellow": return "background-color:#fff5bf; color:#0a0a0a"
        if v == "Red": return "background-color:coral; color:#0a0a0a"
        return ""
    pct_cols = [c for c in df.columns if ("Return" in c) or (c in ["Max Drawdown","Excess Max Drawdown","Expense Ratio","Dividend Yield %"])]
    fmt = {}
    for c in pct_cols:
        if c == "Expense Ratio":
            fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%"
        elif c == "Dividend Yield %":
            fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}%"
        else:
            fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%"
    if "Sortino" in df.columns:
        fmt["Sortino"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    if "Excess Sortino" in df.columns:
        fmt["Excess Sortino"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    styler = df.style.map(color_css, subset=["Color"]).format(fmt)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler

st.sidebar.title("Fund Dashboard")
period_key = st.sidebar.selectbox("Period", ["YTD","1Y","3Y","5Y"], index=1)
mode = st.sidebar.selectbox("View", ["Vs Benchmark","Vs Each Other"], index=0)

if mode == "Vs Benchmark":
    df = build_vs_benchmark(period_key).copy()
    cols = ["Fund","Benchmark","Asset Class","Purpose","Strategy",
            f"Fund Total Return ({period_key})", f"Benchmark Total Return ({period_key})",
            f"Excess Total Return ({period_key})","Excess Sortino","Excess Max Drawdown",
            "Expense Ratio","Dividend Yield %","Points","Color"]
    cols = [c for c in cols if c in df.columns]
    df = df.loc[:, cols] if not df.empty else df
else:
    df = build_vs_each_other(period_key).copy()
    cols = ["Fund","Asset Class","Purpose","Strategy",
            f"Total Return ({period_key})","Sortino","Max Drawdown",
            "Expense Ratio","Dividend Yield %","Points","Color"]
    cols = [c for c in cols if c in df.columns]
    df = df.loc[:, cols] if not df.empty else df

purpose_opts = sorted(df["Purpose"].dropna().unique()) if ("Purpose" in df.columns and not df.empty) else []
asset_opts = sorted(df["Asset Class"].dropna().unique()) if ("Asset Class" in df.columns and not df.empty) else []
purpose_filter = st.sidebar.multiselect("Filter by Purpose", options=purpose_opts, default=[])
asset_filter = st.sidebar.multiselect("Filter by Asset Class", options=asset_opts, default=[])
df_view = df.copy()
if purpose_filter:
    df_view = df_view[df_view["Purpose"].isin(purpose_filter)]
if asset_filter:
    df_view = df_view[df_view["Asset Class"].isin(asset_filter)]

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

st.subheader(mode + f" — {period_key}")
if df_view.empty:
    st.info("No rows for current selection.")
else:
    st.dataframe(style_table(df_view), use_container_width=True)
