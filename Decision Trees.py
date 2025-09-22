#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd

# ----- INPUTS -----
account_type = st.selectbox("Account Type", ["Qualified", "Taxable"])
capital_gains = st.selectbox("Large Capital Gains?", ["Yes", "No"])
purpose = st.selectbox("Purpose", ["Accumulation", "Income"])
distribution = st.selectbox("Need Distributions?", ["Yes", "No"])
age = st.selectbox("Age Range", ["<50", "50-70", "70+"])

# ----- BASE PORTFOLIO LOGIC -----
if purpose == "Accumulation" and age == "<50":
    portfolio = {"US Large Cap": 30, "US SMID Cap": 20, "International": 20, "Growth Themes": 20}
elif purpose == "Accumulation" and age == "50-70":
    portfolio = {"US Large Cap": 25, "US SMID Cap": 15, "International": 15, "Tactical Growth": 20, "Core Bond": 25}
elif purpose == "Accumulation" and age == "70+":
    portfolio = {"OVLH": 34, "KHPI": 33, "PSFF": 33}  # Diversified Protection
elif purpose == "Income" and account_type == "Qualified":
    portfolio = {"KHPI": 30, "OVT": 20, "CLOI": 20, "IGLD": 15, "IQDY": 15}  # Multi-Strategy Income
elif purpose == "Income" and account_type == "Taxable":
    portfolio = {"OVM": 40, "CGSM": 30, "JMST": 10, "SHYD": 10, "HYMB": 10}  # Diversified Muni
else:
    portfolio = {"Custom": 100}

# ----- OVERLAYS -----
portfolio["Denali Structured Return"] = 10
portfolio["Niagara Direct Lending"] = 10

# ----- TAX LOSS HARVESTING -----
if account_type == "Taxable" and capital_gains == "Yes":
    portfolio["Tax Loss Harvesting SMA (Brooklyn/AQR)"] = 20
    # scale down equity slices so total = 100
    equity_keys = [k for k in portfolio.keys() if k not in ["Denali Structured Return","Niagara Direct Lending","Tax Loss Harvesting SMA (Brooklyn/AQR)"]]
    scale_factor = (100 - 10 - 10 - 20) / sum(portfolio[k] for k in equity_keys)
    for k in equity_keys:
        portfolio[k] *= scale_factor

# ----- OUTPUT -----
df = pd.DataFrame(list(portfolio.items()), columns=["Strategy", "Weight (%)"])
st.dataframe(df)
st.bar_chart(df.set_index("Strategy"))

