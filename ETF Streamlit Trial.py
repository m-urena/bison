import numpy as np, pandas as pd, yfinance as yf
from fredapi import Fred
from yahooquery import Ticker as YQT
import streamlit as st
from datetime import date
from functools import lru_cache

st.set_page_config(page_title="Fund Dashboard", layout="wide")

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

@st.cache_data(ttl=3600)
def load_prices(start):
    tickers = sorted(set(list(etf_map.keys()) + [m["benchmark"] for m in etf_map.values()]))
    raw = yf.download(tickers, start=start, end=date.today(), auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Adj Close"].copy()
    else:
        if "Adj Close" not in raw.columns:
            raise KeyError("Adj Close not in yfinance response")
        px = raw[["Adj Close"]].copy()
        px.columns = [tickers[0]]
    px.index = pd.to_datetime(px.index, utc=True, errors="coerce").tz_convert(None)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.loc[:, ~px.columns.duplicated(keep="first")]
    return px

@st.cache_data(ttl=3600)
def load_rf_daily(start):
    try:
        key = "9a093bfd7b591c30fdc29d0d56e1c8f3"
        fred = Fred(api_key=key)
        rf = fred.get_series("DGS1", start).astype(float)/100.0
        rf_df = pd.DataFrame(rf, columns=["RF"]).reindex(pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="B")).ffill()
        rf_daily = (1.0 + rf_df["RF"]).pow(1/252.0) - 1.0
        rf_daily.name = "RF"
        return rf_daily
    except:
        idx = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="B")
        return pd.Series(0.0, index=idx, name="RF")

def sortino_ratio(series, rf_series_daily):
    z = pd.concat([pd.Series(series).dropna(), pd.Series(rf_series_daily).dropna()], axis=1).dropna()
    if z.empty:
        return np.nan
    ex = z.iloc[:,0] - z.iloc[:,1]
    dn = ex[ex < 0].std() * np.sqrt(252)
    if dn == 0 or np.isnan(dn):
        return np.nan
    return round((ex.mean() * 252) / dn,2)

def max_drawdown(r):
    s = pd.Series(r).dropna().astype(float)
    if s.ndim != 1:
        s = s.squeeze()
    if s.empty:
        return np.nan
    w = (1 + s).cumprod()
    return round(float((1 - w.div(w.cummax())).max()),4)

def nearest_on_or_before(idx, ts):
    i = idx.searchsorted(ts, side="right") - 1
    return idx[i] if i >= 0 else None

def nearest_on_or_after(idx, ts):
    i = idx.searchsorted(ts, side="left")
    return idx[i] if i < len(idx) else None

def month_end(dt):
    return (pd.Timestamp(dt).normalize() + pd.offsets.MonthEnd(0))

def period_window(index, period_key):
    index = pd.DatetimeIndex(index).sort_values()
    last = index[-1]

    if period_key == "YTD":
        start_ts = pd.Timestamp(year=last.year, month=1, day=1)
        start = nearest_on_or_after(index, start_ts)
        if start is None:
            start = index[0]
        end = nearest_on_or_before(index, last) or last
        return start, end

    if period_key in ("1Y", "3Y", "5Y"):
        n_years = {"1Y": 1, "3Y": 3, "5Y": 5}[period_key]
        end_me = month_end(last)
        start_me = month_end(pd.Timestamp(end_me) - pd.DateOffset(years=n_years))
        start = nearest_on_or_before(index, start_me) or index[0]
        end = nearest_on_or_before(index, end_me) or last
        return start, end

    return index[0], last


def _to_1d_series(p):
    if isinstance(p, pd.Series):
        return pd.to_numeric(p.dropna(), errors="coerce")

    if isinstance(p, pd.DataFrame):
        if p.shape[1] != 1:
            raise ValueError(f"Expected 1 column, got {p.shape[1]}")
        return pd.to_numeric(p.iloc[:, 0].dropna(), errors="coerce")

    a = np.asarray(p)
    if a.ndim == 1:
        return pd.to_numeric(pd.Series(a).dropna(), errors="coerce")
    if a.ndim == 2 and a.shape[1] == 1:
        return pd.to_numeric(pd.Series(a[:, 0]).dropna(), errors="coerce")
    raise ValueError(f"Expected 1D input, got shape {a.shape}")

def period_return_from_prices(p, start, end):
    s = _to_1d_series(p)
    # guard: ensure datetime index, sorted, unique
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce").tz_convert(None)
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.loc[(s.index >= start) & (s.index <= end)]
    if s.size < 2:
        return np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0])**(1.0/yrs) - 1.0) if yrs >= 1.0 else float(s.iloc[-1] / s.iloc[0] - 1.0)


@lru_cache(maxsize=None)
def get_expense_ratio(ticker):
    if ticker in ["NAGRX","DNLIX"]:
        return 0.0199
    if ticker in ["DFNDX"]:
        return 0.0204
    try:
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

@st.cache_data(ttl=1800)
def build_vs_benchmark(px, rf_daily, period_key, _v=10):
    rows = []
    for fund, meta in etf_map.items():
        bench = meta["benchmark"]
        if fund not in px.columns or bench not in px.columns:
            continue
        if fund == bench:
            s = px.loc[:, fund].dropna()
            if s.shape[0] < 60:
                continue
            start, end = period_window(s.index, period_key)
            f_ret_val = period_return_from_prices(s, start, end)
            b_ret_val = f_ret_val
            ex_ret_val = 0.0 if pd.notna(f_ret_val) else np.nan
            r = s.loc[start:end].pct_change().dropna()
            idx = r.index
            rf = rf_daily.reindex(idx).fillna(0.0)
            f_sort = sortino_ratio(r, rf)
            b_sort = f_sort
            ex_sort = 0.0 if pd.notna(f_sort) else np.nan
            f_dd = max_drawdown(r)
            b_dd = f_dd
            ex_dd = 0.0 if pd.notna(f_dd) else np.nan
        else:
            sf = px.loc[:, fund].dropna()
            sb = px.loc[:, bench].dropna()
            pair_idx = sf.index.intersection(sb.index)
            if pair_idx.size < 60:
                continue
            start, end = period_window(pair_idx, period_key)
            f_ret_val = period_return_from_prices(sf, start, end)
            b_ret_val = period_return_from_prices(sb, start, end)
            ex_ret_val = (f_ret_val - b_ret_val) if pd.notna(f_ret_val) and pd.notna(b_ret_val) else np.nan
            rf_idx = pd.date_range(start=start, end=end, freq="B")
            rf = rf_daily.reindex(rf_idx).fillna(0.0)
            fr = sf.loc[start:end].pct_change().dropna()
            br = sb.loc[start:end].pct_change().dropna()
            idx = fr.index.intersection(br.index).intersection(rf.index)
            fr = fr.loc[idx]
            br = br.loc[idx]
            rf = rf.loc[idx]
            f_sort = sortino_ratio(fr, rf)
            b_sort = sortino_ratio(br, rf)
            ex_sort = (f_sort - b_sort) if pd.notna(f_sort) and pd.notna(b_sort) else np.nan
            f_dd = max_drawdown(fr)
            b_dd = max_drawdown(br)
            ex_dd = (b_dd - f_dd) if pd.notna(f_dd) and pd.notna(b_dd) else np.nan
        rows.append({
            "Fund": fund,
            "Benchmark": bench,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Fund Return": f_ret_val,
            "Benchmark Return": b_ret_val,
            "Excess Return": ex_ret_val,
            "Excess Sortino": ex_sort,
            "Excess Max Drawdown": ex_dd,
            "Expense Ratio": get_expense_ratio(fund),
            "Dividend Yield %": get_dividend_yield(fund)
        })
    df = pd.DataFrame(rows)
    for c in ("Expense Ratio", "Dividend Yield %"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=1800)
def build_vs_each_other(px, rf_daily, period_key,_v=9):
    rows = []
    for fund, meta in etf_map.items():
        if fund not in px.columns:
            continue
        s_px = px.loc[:, [fund]].dropna().iloc[:, 0]
        if s_px.shape[0] < 60:
            continue
        start, end = period_window(s_px.index, period_key)
        ret_val = period_return_from_prices(s_px, start, end)
        r = s_px.loc[start:end].pct_change().dropna()
        rf = rf_daily.reindex(r.index).fillna(0.0)
        rows.append({
            "Fund": fund,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Return": ret_val,
            "Sortino": sortino_ratio(r, rf),
            "Max Drawdown": max_drawdown(r),
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
    df["Color"] = np.select([df["Points"]<=2, df["Points"]<=6], ["Red","Yellow"], default="Green")
    return df

def style_table(df):
    order_map = {"Green":0, "Yellow":1, "Red":2}
    if "Color" in df.columns:
        df = df.assign(_c=df["Color"].map(order_map)).sort_values(["_c","Points"], ascending=[True, False]).drop(columns=["_c"])
    def color_css(v):
        if v == "Green": return "background-color:#d6f5d6; color:#0a0a0a"
        if v == "Yellow": return "background-color:#fff5bf; color:#0a0a0a"
        if v == "Red": return "background-color:coral; color:#0a0a0a"
        return ""
    pct_cols = [c for c in df.columns if c in ["Fund Return","Benchmark Return","Excess Return","Return","Max Drawdown","Excess Max Drawdown"]]
    fmt = {}
    for c in pct_cols:
        fmt[c] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%"
    if "Expense Ratio" in df.columns:
        fmt["Expense Ratio"] = lambda v: "" if pd.isna(v) else f"{float(v)*100:.2f}%"
    if "Dividend Yield %" in df.columns:
        fmt["Dividend Yield %"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}%"
    if "Sortino" in df.columns:
        fmt["Sortino"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    if "Excess Sortino" in df.columns:
        fmt["Excess Sortino"] = lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
    styler = (df.style.map(color_css, subset=["Color"]).format(fmt))
    try:
        styler = styler.hide(axis="index")
    except:
        pass
    return styler

st.sidebar.title("Fund Dashboard")
start_date = st.sidebar.date_input("Start Date", value=date(2015,1,1))
mode = st.sidebar.selectbox("View", ["Vs Benchmark","Vs Each Other"], index=0)
period_key = st.sidebar.selectbox("Period", ["YTD","1Y","3Y","5Y"], index=0)

prices = load_prices(start_date)
rf_daily = load_rf_daily(start_date)

if mode == "Vs Benchmark":
    df = build_vs_benchmark(prices, rf_daily, period_key).copy()
    df = add_bench_points(df)
    cols = ["Fund","Benchmark","Asset Class","Purpose","Strategy","Fund Return","Benchmark Return","Excess Return","Excess Sortino","Excess Max Drawdown","Expense Ratio","Dividend Yield %","Points","Color"]
    df = df.loc[:, [c for c in cols if c in df.columns]]
    view_title = f"Vs Benchmark • {period_key}"
else:
    df = build_vs_each_other(prices, rf_daily, period_key).copy()
    df = add_each_points(df)
    cols = ["Fund","Asset Class","Purpose","Strategy","Return","Sortino","Max Drawdown","Expense Ratio","Dividend Yield %","Points","Color"]
    df = df.loc[:, [c for c in cols if c in df.columns]]
    view_title = f"Vs Each Other • {period_key}"

purpose_opts = sorted(df["Purpose"].dropna().unique()) if "Purpose" in df.columns else []
asset_opts   = sorted(df["Asset Class"].dropna().unique()) if "Asset Class" in df.columns else []
purpose_filter = st.sidebar.multiselect("Filter by Purpose", options=purpose_opts)
asset_filter   = st.sidebar.multiselect("Filter by Asset Class", options=asset_opts)
fund_search    = st.sidebar.text_input("Search Fund (optional)").strip()

df_view = df.copy()
if purpose_filter and "Purpose" in df_view.columns:
    df_view = df_view[df_view["Purpose"].isin(purpose_filter)]
if asset_filter and "Asset Class" in df_view.columns:
    df_view = df_view[df_view["Asset Class"].isin(asset_filter)]
if fund_search and "Fund" in df_view.columns:
    df_view = df_view[df_view["Fund"].str.contains(fund_search, case=False, na=False)]

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

st.subheader(view_title)
st.dataframe(style_table(df_view), use_container_width=True)
