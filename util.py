import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import json

#============
# Json Helper 
#============


# Special Character Handling
def _jd(x): 
    return json.dumps(x, ensure_ascii=False)


#===========
# UI Helper
#===========




#------Portfolio Page------

# Resizing DataFrame Editor on streamlit
def _resize_editor_df(prev: pd.DataFrame, tgt_len: int, times_grid: list[float]) -> pd.DataFrame:
    # Coerce numerics & normalize columns
    if not prev.empty:
        prev = prev.copy()
        for c in ["Time (years)", "Cash Flow", "Rate"]:
            if c not in prev: prev[c] = np.nan
        prev["Time (years)"] = pd.to_numeric(prev["Time (years)"], errors="coerce")
        prev["Cash Flow"]    = pd.to_numeric(prev["Cash Flow"],    errors="coerce")
        prev["Rate"]         = pd.to_numeric(prev["Rate"],         errors="coerce")
    else:
        prev = pd.DataFrame(columns=["Time (years)", "Cash Flow", "Rate"])

    # Truncate or extend to target length
    cur_len = len(prev)
    if cur_len > tgt_len:
        out = prev.iloc[:tgt_len].reset_index(drop=True)
    elif cur_len < tgt_len:
        add = pd.DataFrame({"Time (years)": [np.nan]*(tgt_len-cur_len),
                            "Cash Flow":    [np.nan]*(tgt_len-cur_len),
                            "Rate":         [np.nan]*(tgt_len-cur_len)})
        out = pd.concat([prev, add], ignore_index=True)
    else:
        out = prev.reset_index(drop=True)

    # Apply the time grid (overwrite times to keep them aligned with freq/years)
    if tgt_len > 0:
        out.loc[:tgt_len-1, "Time (years)"] = times_grid
    return out


# Preparing DataFrame to save
def _prep_table_for_save(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Time (years)"] = pd.to_numeric(out["Time (years)"], errors="coerce")
    out["Cash Flow"]    = pd.to_numeric(out["Cash Flow"],    errors="coerce")
    out["Rate"]         = pd.to_numeric(out["Rate"],         errors="coerce")
    # Empty CF → 0
    out["Cash Flow"] = out["Cash Flow"].fillna(0.0)

    # Forward-fill Rate; if the first is NaN → error (cannot infer previous)
    if len(out) > 0 and pd.isna(out.loc[0, "Rate"]):
        raise ValueError("The first period's Rate is empty. Please provide a Rate for period 1.")
    out["Rate"] = out["Rate"].ffill()

    # Guardrail: 1 + Rate > 0 
    if (1.0 + out["Rate"]).le(0.0).any():
        raise ValueError("Invalid Rate: each row must satisfy 1 + Rate > 0.")
    
    #Return Clean df
    return out[["Time (years)", "Cash Flow", "Rate"]]

#Extract (per period) rates from df (with forward filled)
def _build_rates_from_df(df: pd.DataFrame) -> list[float]:
    return [float(r) for r in df["Rate"].tolist()]


# Create a row for portfolio.csv
def _row_from_df(name: str, direction: str, df_payload: pd.DataFrame, years_input, freq_input) -> dict:
    # Years is the **input** years (not the count of rows); periods align via Freq and N
    rates_list = _build_rates_from_df(df_payload)
    df_tcf = df_payload[["Time (years)", "Cash Flow"]]
    return {
        "Name": name,
        "CashFlowType": "Custom",
        "Direction": direction,     # "Positive" or "Negative"
        "Timing": "Immediate",
        "Perpetuity": False,
        "Years": float(years_input),
        "Freq": float(freq_input),
        "PaymentType": "Custom",
        "PaymentParams": df_tcf.to_json(orient="records"),
        "RateType": "Rate Schedule",
        "RateParams": json.dumps({"rates": rates_list}),
        "OptionParams": json.dumps({"has_option": False, "type": None, "schedule": [], "price_basis": None}),
        "Created": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


#Lump Sum payment params builder
def build_lumpsum_payment_params(pmt, yr, years):
    i = 0
    rows = []
    if years < yr:
        raise ValueError("Horizon Year must be greater than payment year")

    while i <= years:
        if i != yr:
            rows.append({
                "Time (years)": i,
                "Cash Flow": 0
            })
        else:
            rows.append({
                "Time (years)": i,
                "Cash Flow": pmt
            })
        i += 1
    return json.dumps(rows) 

# Annuity + Final Payout param builder
def build_final_payment_params(pmt, fin_pmt, years, freq, timing):
    years = float(years)
    freq = int(freq)

    # start time based on timing convention
    if timing == "Immediate":
        i = 1 / freq
    elif timing == "Due":
        i = 0.0
    else:
        i = 0.0  # default fallback

    rows = []
    while round(i, 10) < years:
        rows.append({
            "Time (years)": round(i, 10),
            "Cash Flow": pmt
        })
        i += 1 / freq

    # add final payment (avoid duplication)
    if not rows or rows[-1]["Time (years)"] != years:
        rows.append({
            "Time (years)": years,
            "Cash Flow": fin_pmt
        })

    return json.dumps(rows)