import numpy as np, pandas as pd, yfinance as yf
import streamlit as st
from datetime import date
from functools import lru_cache

st.set_page_config(page_title="Fund Dashboard", layout="wide")

try:
    import mstarpy as ms
    MSTARPY_AVAILABLE = True
except Exception:
    MSTARPY_AVAILABLE = False

try:
    from fredapi import Fred
except Exception:
    Fred = None

try:
    from yahooquery import Ticker as YQT
except Exception:
    YQT = None

etf_map = {
    "IBIT":  {"benchmark": "IBIT", "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Thematic"},
    "IQDY":  {"benchmark": "ACWX", "asset_class": "Equity",       "purpose": "Income",       "strategy": "Foreign"},
    "QQQ":   {"benchmark": "QQQ",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Growth"},
    "DNLIX": {"benchmark": "SPY",  "asset_class": "Alts",         "purpose": "Preservation", "strategy": "Hedged"},
    "AVUV":  {"benchmark": "IJR",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Small Cap"},
    "GRID":  {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Thematic"},
    "XMMO":  {"benchmark": "IJH",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Growth"},
    "PAVE":  {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Thematic"},
    "OVF":   {"benchmark": "ACWX", "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Foreign"},
    "SCHD":  {"benchmark": "IWD",  "asset_class": "Equity",       "purpose": "Income",       "strategy": "Dividend"},
    "OVLH":  {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Hedged"},
    "DGRW":  {"benchmark": "SCHD", "asset_class": "Equity",       "purpose": "Income",       "strategy": "Dividend"},
    "FLQM":  {"benchmark": "IJH",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Mid Cap"},
    "KHPI":  {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Hedged"},
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
    "IAU":   {"benchmark": "GLD",  "asset_class": "Alts",         "purpose": "Preservation", "strategy": "Commodity"},
    "IGLD":  {"benchmark": "GLD",  "asset_class": "Alts",         "purpose": "Preservation", "strategy": "Commodity"},
    "IEI":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "NAGRX": {"benchmark": "AGG",  "asset_class": "Alts",         "purpose": "Preservation", "strategy": "Core Bond"},
    "IWF":   {"benchmark": "IWF",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Growth"},
    "OVS":   {"benchmark": "IJR",  "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Small Cap"},
    "OVL":   {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Large Cap"},
    "OVM":   {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Municipal"},
    "CLOI":  {"benchmark": "BKLN", "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Alt Credit"},
    "FIW":   {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Thematic"},
    "PEY":   {"benchmark": "IJH",  "asset_class": "Equity",       "purpose": "Income",       "strategy": "Dividend"},
    "GSIMX": {"benchmark": "ACWX", "asset_class": "Equity",       "purpose": "Accumulation", "strategy": "Foreign"},
    "DFNDX": {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Preservation", "strategy": "Hedged"},
    "PSFF":  {"benchmark": "SPY",  "asset_class": "Equity",       "purpose": "Income",       "strategy": "Hedged"},
    "CPITX": {"benchmark": "HYG",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"}
}

def nearest_on_or_after(idx, ts):
    i = pd.DatetimeIndex(idx).searchsorted(ts, side="left")
    return idx[i] if i < len(idx) else None

def nearest_on_or_before(idx, ts):
    i = pd.DatetimeIndex(idx).searchsorted(ts, side="right") - 1
    return idx[i] if i >= 0 else None

def window_for_period(index, period_key):
    index = pd.DatetimeIndex(index).sort_values()
    end = index[-1]
    if period_key == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif period_key == "1Y":
        start = end - pd.Timedelta(days=365)
    elif period_key == "3Y":
        start = end - pd.Timedelta(days=3*365)
    elif period_key == "5Y":
        start = end - pd.Timedelta(days=5*365)
    else:
        start = index[0]
    s = nearest_on_or_after(index, start) or index[0]
    e = nearest_on_or_before(index, end) or end
    return s, e

@st.cache_data(ttl=3600)
def load_close_prices(period_key):
    today = pd.Timestamp.today().normalize()
    span = {"YTD": 400, "1Y": 450, "3Y": 3*370+30, "5Y": 5*370+30}.get(period_key, 5*370+30)
    start = (today - pd.Timedelta(days=span)).date()
    tickers = sorted(set(list(etf_map.keys()) + [m["benchmark"] for m in etf_map.values()]))
    raw = yf.download(tickers, start=start, end=date.today(), auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise KeyError("Close not in yfinance response")
        px = raw[["Close"]].copy()
        px.columns = [tickers[0]]
    px.index = pd.to_datetime(px.index, utc=True, errors="coerce").tz_convert(None)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.loc[:, ~px.columns.duplicated(keep="first")]
    return px

@st.cache_data(ttl=3600)
def load_dividends_map(period_key):
    today = pd.Timestamp.today().normalize()
    span = {"YTD": 400, "1Y": 450, "3Y": 3*370+30, "5Y": 5*370+30}.get(period_key, 5*370+30)
    start = (today - pd.Timedelta(days=span))
    dmap = {}
    for t in set(list(etf_map.keys()) + [m["benchmark"] for m in etf_map.values()]):
        try:
            divs = yf.Ticker(t).dividends
            if divs is None or divs.empty:
                dmap[t] = pd.Series(dtype=float)
                continue
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)
            dmap[t] = pd.to_numeric(divs[divs.index >= start], errors="coerce").dropna()
        except:
            dmap[t] = pd.Series(dtype=float)
    return dmap

def build_total_return_series_from_yahoo(price_series, dividend_series):
    s = pd.to_numeric(pd.Series(price_series).dropna(), errors="coerce").dropna()
    d = pd.to_numeric(pd.Series(dividend_series).dropna(), errors="coerce") if dividend_series is not None else pd.Series(dtype=float)
    d = d.reindex(s.index).fillna(0.0)
    pr = s.pct_change().fillna(0.0)
    add = (d / s.shift(1)).fillna(0.0)
    tr = (1.0 + pr + add).cumprod()
    tr.iloc[0] = 1.0
    return tr

def period_return_and_daily(tr, start, end):
    tr = tr.loc[(tr.index >= start) & (tr.index <= end)]
    if tr.size < 2:
        return np.nan, None
    days = (tr.index[-1] - tr.index[0]).days
    total = float(tr.iloc[-1] / tr.iloc[0] - 1.0)
    ann = float((tr.iloc[-1] / tr.iloc[0])**(365.25/days) - 1.0) if days >= 365 else total
    r_daily = tr.pct_change().dropna()
    return ann, r_daily

@st.cache_data(ttl=3600)
def load_rf_daily(period_key):
    today = pd.Timestamp.today().normalize()
    span = {"YTD": 400, "1Y": 450, "3Y": 3*370+30, "5Y": 5*370+30}.get(period_key, 5*370+30)
    start = (today - pd.Timedelta(days=span)).date()
    try:
        if Fred is None:
            raise RuntimeError("No fredapi")
        fred_key = st.secrets.get("FRED_API_KEY", None)
        fred = Fred(api_key=fred_key) if fred_key else Fred()
        rf = fred.get_series("DGS1", start).astype(float)/100.0
        rf_df = pd.DataFrame(rf, columns=["RF"]).reindex(pd.date_range(start=start, end=today, freq="B")).ffill()
        rf_daily = (1.0 + rf_df["RF"]).pow(1/252.0) - 1.0
        rf_daily.name = "RF"
        return rf_daily
    except Exception:
        idx = pd.date_range(start=start, end=today, freq="B")
        return pd.Series(0.0, index=idx, name="RF")

def sortino_ratio(series_daily, rf_daily):
    z = pd.concat([pd.Series(series_daily).dropna(), pd.Series(rf_daily)], axis=1).dropna()
    if z.empty:
        return np.nan
    ex = z.iloc[:,0] - z.iloc[:,1]
    dn = ex[ex < 0].std() * np.sqrt(252)
    if dn == 0 or np.isnan(dn):
        return np.nan
    return round((ex.mean() * 252) / dn, 2)

def max_drawdown(series_daily):
    s = pd.Series(series_daily).dropna().astype(float)
    if s.empty:
        return np.nan
    w = (1 + s).cumprod()
    return round(float((1 - w.div(w.cummax())).max()), 4)

@lru_cache(maxsize=None)
def get_expense_ratio(ticker):
    if ticker in ["NAGRX","DNLIX"]:
        return 0.0199
    if ticker in ["DFNDX"]:
        return 0.0204
    try:
        if YQT is not None:
            yq = YQT(ticker)
            v = None
            prof = yq.fund_profile
            if isinstance(prof, dict) and ticker in prof:
                fees = prof[ticker].get("feesExpensesInvestment") or prof[ticker].get("feesExpensesOperating") or {}
                for k in ("annualReportExpenseRatio","netExpRatio","grossExpRatio","expenseRatio"):
                    if fees.get(k) is not None:
                        v = fees[k]; break
            if v is None:
                sd = yq.summary_detail
                if isinstance(sd, dict) and ticker in sd:
                    for k in ("annualReportExpenseRatio","expenseRatio"):
                        if sd[ticker].get(k) is not None:
                            v = sd[ticker][k]; break
            if v is not None:
                v = float(v)
                if v >= 0.05 or v > 5:
                    v = v / 100.0
                return round(v, 6)
        yi = yf.Ticker(ticker).info or {}
        v = yi.get("expenseRatio")
        if v is None:
            return np.nan
        v = float(v)
        if v >= 0.05 or v > 5:
            v = v / 100.0
        return round(v, 6)
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
            return round(float(y) * 100, 2)
        divs = t.dividends
        if divs is None or divs.empty:
            return 0
        if getattr(divs.index, "tz", None) is not None:
            divs.index = divs.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        last12 = divs[divs.index >= cutoff]
        if last12.empty:
            return 0
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
        return round((total / float(price)) * 100, 2)
    except Exception:
        return np.nan

def get_mstar_total_return_series(ticker, period_key):
    if not MSTARPY_AVAILABLE:
        return None
    try:
        user = st.secrets.get("MSTAR_USERNAME", None)
        pwd  = st.secrets.get("MSTAR_PASSWORD", None)
        api  = st.secrets.get("MSTAR_API_KEY", None)
        if user and pwd and hasattr(ms, "auth"):
            try:
                ms.auth.login(user, pwd)
            except Exception:
                pass
        f = ms.Funds(ticker)
        # Expect a DataFrame or list of dicts with date + total return index or daily total return %
        # Try the most common accessors; fallback to None to trigger Yahoo TRI
        df = None
        for attr in ("nav", "performance", "returns", "total_return"):
            if hasattr(f, attr):
                try:
                    out = getattr(f, attr)()
                    if out is not None:
                        df = pd.DataFrame(out)
                        break
                except Exception:
                    continue
        if df is None or df.empty:
            return None
        # Normalize
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
        # We expect either a level/ index (TRI) or a daily total return % column
        tri = None
        if "totalReturn" in df.columns:
            # If totalReturn appears cumulative index (e.g., 230.xx) normalize to base=1
            s = pd.to_numeric(df["totalReturn"], errors="coerce").dropna()
            if s.max() > 10:  # likely an index level rather than % daily
                tri = (s / s.iloc[0]).rename(ticker)
            else:
                # percentage daily? convert to TRI
                tri = (1.0 + s/100.0).cumprod().rename(ticker)
        elif "tri" in df.columns:
            s = pd.to_numeric(df["tri"], errors="coerce").dropna()
            tri = (s / s.iloc[0]).rename(ticker)
        elif "nav" in df.columns and "distribution" in df.columns:
            px = pd.to_numeric(df["nav"], errors="coerce")
            dv = pd.to_numeric(df["distribution"], errors="coerce")
            tri = build_total_return_series_from_yahoo(px, dv).rename(ticker)
        if tri is None or tri.empty:
            return None
        tri.index = pd.to_datetime(tri.index, utc=True, errors="coerce").tz_convert(None)
        tri = tri[~tri.index.duplicated(keep="last")].sort_index()
        return tri
    except Exception:
        return None

def get_total_return_series(ticker, period_key, px=None, dmap=None):
    tri = get_mstar_total_return_series(ticker, period_key)
    if tri is not None and not tri.empty:
        return tri
    if px is None or dmap is None or ticker not in px.columns:
        return None
    return build_total_return_series_from_yahoo(px[ticker], dmap.get(ticker, None)).rename(ticker)

@st.cache_data(ttl=900)
def build_vs_benchmark(period_key):
    px = load_close_prices(period_key)
    dmap = load_dividends_map(period_key)
    rf_daily = load_rf_daily(period_key)
    rows = []
    for fund, meta in etf_map.items():
        bench = meta["benchmark"]
        if fund not in px.columns or bench not in px.columns:
            continue
        ftr = get_total_return_series(fund, period_key, px, dmap)
        btr = get_total_return_series(bench, period_key, px, dmap)
        if ftr is None or btr is None or ftr.empty or btr.empty:
            continue
        idx = ftr.index.intersection(btr.index)
        if idx.size < 60:
            continue
        start, end = window_for_period(idx, period_key)
        f_ann, f_daily = period_return_and_daily(ftr.loc[idx], start, end)
        b_ann, b_daily = period_return_and_daily(btr.loc[idx], start, end)
        if f_daily is None or b_daily is None:
            continue
        rfi = rf_daily.reindex(f_daily.index).fillna(0.0)
        fs = sortino_ratio(f_daily, rfi)
        bs = sortino_ratio(b_daily, rfi)
        fdd = max_drawdown(f_daily)
        bdd = max_drawdown(b_daily)
        rows.append({
            "Fund": fund,
            "Benchmark": bench,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Fund Return": f_ann,
            "Benchmark Return": b_ann,
            "Excess Return": (f_ann - b_ann) if pd.notna(f_ann) and pd.notna(b_ann) else np.nan,
            "Excess Sortino": (fs - bs) if pd.notna(fs) and pd.notna(bs) else np.nan,
            "Excess Max Drawdown": (bdd - fdd) if pd.notna(fdd) and pd.notna(bdd) else np.nan,
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=900)
def build_vs_each_other(period_key):
    px = load_close_prices(period_key)
    dmap = load_dividends_map(period_key)
    rf_daily = load_rf_daily(period_key)
    rows = []
    for fund, meta in etf_map.items():
        tri = get_total_return_series(fund, period_key, px, dmap)
        if tri is None or tri.empty:
            continue
        start, end = window_for_period(tri.index, period_key)
        ann, daily = period_return_and_daily(tri, start, end)
        if daily is None:
            continue
        rfi = rf_daily.reindex(daily.index).fillna(0.0)
        rows.append({
            "Fund": fund,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Return": ann,
            "Sortino": sortino_ratio(daily, rfi),
            "Max Drawdown": max_drawdown(daily),
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    return pd.DataFrame(rows)

def quartile_points(s):
    r = pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")
    return pd.cut(r, bins=[0,0.25,0.5,0.75,1.0000001], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)

def add_bench_points(df):
    er = pd.to_numeric(df.get("Excess Return"), errors="coerce")
    es = pd.to_numeric(df.get("Excess Sortino"), errors="coerce")
    dy = pd.to_numeric(df.get("Dividend Yield %"), errors="coerce")/100.0
    pts = ((er > -0.01).astype(int) + (es > -0.05).astype(int) + (dy > 0.025).astype(int)).fillna(0).astype(int)
    df["Points"] = pts
    df["Color"] = np.select([df["Points"]>=2, df["Points"]==1], ["Green","Yellow"], default="Red")
    return df

def add_each_points(df):
    p = quartile_points(df.get("Return")) + quartile_points(df.get("Sortino")) + quartile_points(df.get("Dividend Yield %"))
    df["Points"] = p.astype(int)
    df["Color"] = np.select([df["Points"]<=2, df["Points"]<=6], ["Red","Yellow
