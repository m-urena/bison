import numpy as np, pandas as pd, yfinance as yf
from fredapi import Fred
from yahooquery import Ticker
import streamlit as st
from datetime import date


st.set_page_config(page_title="Fund Dashboard", layout="wide")

etf_map = {
    "IBIT":  {"benchmark": "IBIT", "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "IQDY":  {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Income",       "strategy": "Foreign"},
    "QQQ":   {"benchmark": "QQQ",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "DNLIX": {"benchmark": "SPY",  "asset_class": "Alt", "purpose": "Preservation", "strategy": "Hedged"},
    "AVUV":  {"benchmark": "IJR",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Small Cap"},
    "GRID":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "XMMO":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Growth"},
    "PAVE":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Thematic"},
    "OVF":   {"benchmark": "ACWX", "asset_class": "Equity", "purpose": "Preservation", "strategy": "Foreign"},
    "SCHD":  {"benchmark": "IWD",  "asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"}, 
    "OVLH":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "DGRW":  {"benchmark": "SCHD",  "asset_class": "Equity", "purpose": "Income",       "strategy": "Dividend"},#was compared with SPY, changed to SCHD given they do the same thing
    "FLQM":  {"benchmark": "IJH",  "asset_class": "Equity", "purpose": "Accumulation", "strategy": "Mid Cap"},
    "KHPI":  {"benchmark": "SPY",  "asset_class": "Equity", "purpose": "Preservation", "strategy": "Hedged"},
    "IEF":   {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Treasury"},
    "ICSH":  {"benchmark": "BIL","asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
    "CGSM":  {"benchmark": "MUB",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "Municipal"},
    "SHYD":  {"benchmark": "HYD",  "asset_class": "Fixed Income", "purpose": "Income",       "strategy": "High Yield"},
    "BIL":   {"benchmark": "BIL","asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Cash"},
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
    "NAGRX": {"benchmark": "AGG",  "asset_class": "Fixed Income", "purpose": "Preservation", "strategy": "Core Bond"},
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

@st.cache_data(ttl=3600)
def load_prices(start):
    tickers = sorted(set(list(etf_map.keys()) + [m["benchmark"] for m in etf_map.values()]))
    px = yf.download(tickers, start=start, end=date.today(), auto_adjust=False, progress=False)["Adj Close"]#change to Close for price return not total
    px.index = pd.to_datetime(px.index, utc=True, errors="coerce").tz_convert(None)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    return px

@st.cache_data(ttl=3600)
def load_rf_daily(start):
    try:
        from fredapi import Fred
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
    z = pd.concat([pd.Series(series).dropna(), rf_series_daily], axis=1).dropna()
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

from functools import lru_cache

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

        # Normalize units:
        # - APIs sometimes return percent (e.g., 0.19 for 0.19%) or fraction (0.0019)
        # - Heuristic: if >= 0.05 (>=5%), it’s almost certainly a percent → divide by 100
        #   and also guard absurd values (>5) by dividing by 100.
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
        # Otherwise compute trailing 12-month yield from dividends / latest price
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
        return round((total / float(price)) * 100, 2)

    except Exception:
        return np.nan

@st.cache_data(ttl=1800)
def build_vs_benchmark(px, rets, rf_daily):
    rows = []
    for etf, meta in etf_map.items():
        bench = meta["benchmark"]
        if etf not in rets.columns or bench not in rets.columns:
            continue
        z = rets.loc[:, [etf, bench]].dropna()
        if z.shape[0] < 60:
            continue
        etf_ret = z.iloc[:, 0]
        bench_ret = z.iloc[:, 1]
        etf_ann = float(etf_ret.mean() * 252)
        bench_ann = float(bench_ret.mean() * 252)
        ex_ret_ann = float((etf_ret - bench_ret).mean() * 252)
        etf_sort = sortino_ratio(etf_ret, rf_daily)
        bench_sort = sortino_ratio(bench_ret, rf_daily)
        ex_sort = etf_sort - bench_sort if pd.notna(etf_sort) and pd.notna(bench_sort) else np.nan
        etf_dd = max_drawdown(etf_ret)
        bench_dd = max_drawdown(bench_ret)
        ex_dd = bench_dd - etf_dd if pd.notna(etf_dd) and pd.notna(bench_dd) else np.nan
        exp_ratio = get_expense_ratio(etf)
        dy = get_dividend_yield(etf)
        rows.append({
            "Fund": etf,
            "Benchmark": bench,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Fund Total Return (annualized)": etf_ann,
            "Benchmark Total Return (annualized)": bench_ann,
            "Excess Total Return (annualized)": ex_ret_ann,
            "Excess Sortino": ex_sort,
            "Excess Max Drawdown": ex_dd,
            "Expense Ratio": exp_ratio,
            "Dividend Yield %": dy
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def build_vs_each_other_simple(rets, rf_daily):
    rows = []
    for etf, meta in etf_map.items():
        if etf not in rets.columns:
            continue
        r = rets.loc[:, [etf]].dropna().iloc[:, 0]
        if r.shape[0] < 60:
            continue
        ann_ret = float(r.mean() * 252)
        sr = sortino_ratio(r, rf_daily)
        mdd = max_drawdown(r)
        exp_ratio = get_expense_ratio(etf)
        dy = get_dividend_yield(etf)
        rows.append({
            "Fund": etf,
            "Asset Class": meta["asset_class"],
            "Purpose": meta["purpose"],
            "Strategy": meta["strategy"],
            "Total Return (annualized)": ann_ret,
            "Sortino": sr,
            "Max Drawdown": mdd,
            "Expense Ratio": exp_ratio,
            "Dividend Yield %": dy
        })
    return pd.DataFrame(rows)

def add_bench_points(df):
    er = pd.to_numeric(df.get("Excess Total Return (annualized)"), errors="coerce")
    es = pd.to_numeric(df.get("Excess Sortino"), errors="coerce")
    dy = pd.to_numeric(df.get("Dividend Yield %"), errors="coerce")/100.0
    pts = ((er > -0.01).astype(int) + (es > -0.05).astype(int) + (dy > 0.025).astype(int)).fillna(0).astype(int)
    df["Points"] = pts
    df["Color"] = np.select([df["Points"]>=2, df["Points"]==1], ["Green","Yellow"], default="Red")
    return df

def quartile_points(s):
    r = pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")
    return pd.cut(r, bins=[0,0.25,0.5,0.75,1.0000001], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)

def add_each_points(df):
    p = quartile_points(df.get("Total Return (annualized)")) + quartile_points(df.get("Sortino")) + quartile_points(df.get("Dividend Yield %"))
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

    pct_cols = [c for c in df.columns if c in [
        "Fund Total Return (annualized)",
        "Benchmark Total Return (annualized)",
        "Excess Total Return (annualized)",
        "Total Return (annualized)",
        "Max Drawdown",
        "Excess Max Drawdown",
    ]]
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
start_date = st.sidebar.date_input("Start Date", value=date(2020,1,1))
mode = st.sidebar.selectbox("View", ["Vs Benchmark","Vs Each Other"], index=0)

prices = load_prices(start_date)
rf_daily = load_rf_daily(start_date)
common_idx = prices.index.intersection(rf_daily.index)
prices = prices.loc[common_idx]
rf_daily = rf_daily.loc[common_idx]
rets = prices.pct_change().dropna()

if mode == "Vs Benchmark":
    df = build_vs_benchmark(prices, rets, rf_daily).copy()
    df = add_bench_points(df)
    cols = ["Fund","Benchmark","Asset Class","Purpose","Strategy","Fund Total Return (annualized)","Benchmark Total Return (annualized)","Excess Total Return (annualized)","Excess Sortino","Excess Max Drawdown","Expense Ratio","Dividend Yield %","Points","Color"]
    cols = [c for c in cols if c in df.columns]
    df = df.loc[:, cols]
    st.subheader("Vs Benchmark")
    st.dataframe(style_table(df), use_container_width=True)
else:
    df = build_vs_each_other_simple(rets, rf_daily).copy()
    df = add_each_points(df)
    cols = ["Fund","Asset Class","Purpose","Strategy","Total Return (annualized)","Sortino","Max Drawdown","Expense Ratio","Dividend Yield %","Points","Color"]
    cols = [c for c in cols if c in df.columns]
    df = df.loc[:, cols]
    st.subheader("Vs Each Other")
    st.dataframe(style_table(df), use_container_width=True)
