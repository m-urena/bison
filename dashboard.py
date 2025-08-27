#!/usr/bin/env python
# coding: utf-8

# In[28]:


import yfinance as yf
import numpy as np
import pandas as pd
from fredapi import Fred
import warnings; warnings.simplefilter('ignore')
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LinearRegression
from yahooquery import Ticker
import streamlit as st


# In[26]:


etf_benchmark_map = {
    #Equities
    "OVL": "SPY",
    "QQQ": "SPY",
    "IWF": "IWF",
    "SCHD": "IWD",
    "DGRW": "SPY",
    "PEY": "SPY",
    "XMMO": "IJH",
    "PAVE": "SPY",
    "FIW": "SPY",
    "CIBR": "SPY",
    "GRID": "SPY",
    "AVUV": "IJR",
    "OVS": "IJR",
    "FLQM": "IJH",
    "OVF": "ACWX",
    "GSIMX": "ACWX",
    "OVLH": "SPY",
    "PSFF": "SPY",
    "IQDY": "ACWX",
    #Fixed Income
    "BIL": "AGG",
    "ICSH": "AGG",
    "SHY": "AGG",
    "OVT": "SPSB",
    "IEF": "AGG",
    "OVB": "AGG",
    "MBSF": "MBB",
    "CLOI": "BKLN",
    "CLOB": "BKLN",
    "IEI": "AGG",
    "CPITX": "HYG",
    "ESIIX": "HYG",
    "OVM": "MUB",
    "CGSM": "MUB",
    "JMST": "MUB",
    "SHYD": "HYD",
    "HYMB": "HYD",
    #Alts
    "NAGRX": "AGG",
    "DNLIX": "AGG",
    "DFNDX": "SPY",
    "IAU": "GLD",
    "IGLD": "GLD",
    "IBIT": "SPY",
    "KHPI": "SPY"
}

fred_key = "9a093bfd7b591c30fdc29d0d56e1c8f3"
fred = Fred(api_key=fred_key)
start_date="2025-03-01"
rf_series = fred.get_series("DGS1", start=start_date)
rf_series = rf_series / 100.0
rf_df = pd.DataFrame(rf_series, columns=["RF"])
rf_df = rf_df.reindex(pd.date_range(start=start_date, end=pd.Timestamp.today()), method="ffill")

all_tickers = list(set(etf_benchmark_map.keys()) | set(etf_benchmark_map.values()))
prices = yf.download(all_tickers, start=start_date, auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

rf_aligned = rf_df.reindex(returns.index, method="ffill")["RF"]

def calc_beta(etf_ret, bench_ret):
    if etf_ret.equals(bench_ret): 
        return 1.0
    X = bench_ret.values.reshape(-1,1)
    y = etf_ret.values
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def sortino_ratio(series, rf_series):
    excess = series - rf_series/252
    downside = excess[excess < 0].std() * np.sqrt(252)
    if downside == 0:
        return np.nan
    return (excess.mean() * 252) / downside

def get_expense_ratio(ticker):
    if ticker in ["NAGRX", "DNLIX"]:
        return 0.0199  # 1.99%
    t = Ticker(ticker)
    profile = t.fund_profile
    if ticker in profile:
        fees = profile[ticker].get("feesExpensesInvestment") or profile[ticker].get("feesExpensesOperating")
        if fees:
            return fees.get("annualReportExpenseRatio")
    return None
    
adv = yf.download(all_tickers, start=start_date, auto_adjust=True)["Volume"]
adv_mean = adv.mean()

def max_drawdown(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def get_dividend_yield(ticker):
    t = yf.Ticker(ticker)
    divs = t.dividends

    if divs.empty:
        yld = t.info.get("trailingAnnualDividendYield")
        if yld:
            return round(yld * 100, 2)
        return None

    hist = t.history(period="5d")
    if hist.empty:
        price = t.info.get("regularMarketPrice") or t.info.get("previousClose")
    else:
        price = hist["Close"].dropna().iloc[-1]

    if not price or price == 0 or pd.isna(price):
        return None

    one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
    last12m_divs = divs[divs.index >= one_year_ago]

    if last12m_divs.empty:
        yld = t.info.get("trailingAnnualDividendYield")
        return round(yld * 100, 2) if yld else None

    total_divs = last12m_divs.sum()

    if len(last12m_divs) >= 4:
        avg_quarterly = last12m_divs.tail(4).mean()
        total_divs = avg_quarterly * 4

    return round((total_divs / price) * 100, 2)


momentum_6m = prices.pct_change(120).iloc[-1] 
results = []
for etf, bench in etf_benchmark_map.items():
    if etf in returns and bench in returns:
        etf_ret = returns[etf]
        bench_ret = returns[bench]

        excess_return = (etf_ret - bench_ret).mean() * 252
        etf_sortino = sortino_ratio(etf_ret, rf_aligned)
        bench_sortino = sortino_ratio(bench_ret, rf_aligned)
        excess_sortino = etf_sortino - bench_sortino
        mom6m = momentum_6m.get(etf, np.nan)
        expense_ratio = get_expense_ratio(etf)
        avg_daily_vol_M = adv_mean.get(etf, np.nan) / 1_000_000
        etf_dd = max_drawdown(etf_ret)
        bench_dd = max_drawdown(bench_ret)
        rel_dd = etf_dd - bench_dd   
        div_yield=get_dividend_yield(etf)
        
        results.append({
            "ETF": etf,
            "Benchmark": bench,
            "Excess Return(annualized)": excess_return,
            "Excess Sortino": excess_sortino,
            "6 Month Momentum": mom6m,
            "Expense Ratio": expense_ratio,
            "Average Volume(M)": avg_daily_vol_M,
            "Excess Max Drawdown": rel_dd,
            "Dividend Yield %":div_yield
        })

results_df = pd.DataFrame(results)


# In[23]:


print(results_df.head())


# In[27]:


metrics = [
    "Excess Return(annualized)",
    "Excess Sortino",
    "6 Month Momentum",
    "Expense Ratio",
    "Excess Max Drawdown",
    "Average Volume(M)",
    "Dividend Yield %"
]

for col in metrics:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

def score_quartiles(series, higher_is_better=True):
    try:
        if higher_is_better:
            scored = pd.qcut(series.rank(method="first"), 4, labels=[0,1,2,3])
        else:
            scored = pd.qcut((-series).rank(method="first"), 4, labels=[0,1,2,3])
        return scored.astype(float).fillna(0).astype(int)
    except ValueError:
        return pd.Series([0]*len(series), index=series.index)

results_df["Excess Return Score"]   = score_quartiles(results_df["Excess Return(annualized)"], higher_is_better=True)
results_df["Excess Sortino Score"]  = score_quartiles(results_df["Excess Sortino"], higher_is_better=True)
results_df["Momentum Score"]        = score_quartiles(results_df["6 Month Momentum"], higher_is_better=True)
results_df["Expense Ratio Score"]   = score_quartiles(results_df["Expense Ratio"], higher_is_better=False)
results_df["Drawdown Score"]        = score_quartiles(results_df["Excess Max Drawdown"], higher_is_better=False)
results_df["Weighted Volume Score"]          = (score_quartiles(results_df["Average Volume(M)"], higher_is_better=True)/3).round(0).astype(int)
results_df["Yield Score"] = score_quartiles(results_df["Dividend Yield %"], higher_is_better=True)


results_df["Total Score"] = np.ceil(
    2*results_df["Excess Return Score"] 
    + 1.5*results_df["Excess Sortino Score"] 
    + results_df["Momentum Score"]
    + results_df["Expense Ratio Score"]
    + results_df["Drawdown Score"]
    + results_df["Weighted Volume Score"]
     + results_df["Yield Score"]
).astype(int)

def categorize(score):
    if score > 12:
        return "Green"
    elif score >= 6:
        return "Yellow"
    else:
        return "Red"

results_df["Category"] = results_df["Total Score"].apply(categorize)

def highlight_category(val):
    color_map = {"Green": "background-color: lightgreen",
                 "Yellow": "background-color: khaki",
                 "Red": "background-color: lightcoral"}
    return color_map.get(val, "")

results_df = results_df.sort_values("Total Score", ascending=False)
styled = results_df.style.applymap(highlight_category, subset=["Category"])
styled


# In[32]:


st.set_page_config(page_title="ETF Dashboard", layout="wide")

st.title("ETF Dashboard")

# put your DataFrame + scoring code above this
st.dataframe(styled)

