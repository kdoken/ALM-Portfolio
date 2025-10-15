import streamlit as st
import matplotlib.pyplot as plt
import re, numpy as np
import pandas as pd
from datetime import datetime as dt
import json

from functions import (
    accu_com,
    pv_lump,
    fv_lump,
    annuity,
    pv_cashflows,
    fv_cashflows,
    make_delta_func,
    analyze_portfolio_durations
)

from util import (
    _jd,
    _resize_editor_df,
    _prep_table_for_save,
    _row_from_df,
    _build_rates_from_df,
    build_lumpsum_payment_params,
    build_final_payment_params
)

st.set_page_config(page_title="Finance Calculator", layout="wide")

page = st.sidebar.radio(
    "Calculator", ["Time Value (Single Sum)", "Annuity", "Portfolio", "ALM" ]
)
# ——————————————————————————
# Time Value of Money (single cash flow)
# ——————————————————————————
if page == "Time Value (Single Sum)":
    st.header("Time Value of Money")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        pv = st.number_input("Present Value (leave 0 if unknown)", value=0.0, step=0.01)
        fv = st.number_input("Future Value (leave 0 if unknown)", value=0.0, step=0.01)
        t = st.number_input("t (periods/years)", value=1.0, min_value=0.0, step=1.0)
        freq = 1  # treat t as years unless nominal compounding overrides it

    # Pick a rate model
    rate_mode = st.radio(
        "Rate type",
        [
            "Effective Interest",
            "Simple Interest",
            "Nominal Interest",
            "Continuous Force of Interest",
            "Rate Schedule",
            "Time Varying Continuous Interest",
        ],
        horizontal=True,
    )

    # Map UI into accu_com kwargs
    acc_kwargs = {}
    if rate_mode == "Effective Interest":
        eff_i = st.number_input("i (effective per period)", value=0.05, step=0.001)
        acc_kwargs["eff_i"] = eff_i

    elif rate_mode == "Simple Interest":
        simple_i = st.number_input("i (per period)", value=0.05, step=0.001)
        acc_kwargs["simple_i"] = simple_i

    elif rate_mode == "Nominal Interest":
        nom_i = st.number_input("j (nominal per year)", value=0.06, step=0.001)
        m = st.number_input("m (compounds/year)", value=12, min_value=1, step=1)
        acc_kwargs["nom_i"] = nom_i
        acc_kwargs["freq"] = int(m)  # pass compounding frequency to accu_com

    elif rate_mode == "Rate Schedule":
        # Use integer periods for the schedule; round t if needed
        n = int(max(0, round(t)))
        st.caption(f"Using n = {n} period(s) from t = {t} for the schedule.")
        if n == 0:
            st.warning("Increase t to at least 1 to enter a schedule.")
            acc_kwargs["rate_sch"] = []
        else:
            st.write("Enter the effective rate per period (decimal).")
            rates = []
            cols_per_row = 4
            for k in range(1, n + 1):
                if (k - 1) % cols_per_row == 0:
                    row = st.columns(cols_per_row)
                col = row[(k - 1) % cols_per_row]
                with col:
                    r_k = st.number_input(
                        f"i[{k}]",
                        value=0.05,
                        step=0.001,
                        format="%.6f",
                        key=f"rate_sch_{k}",
                    )
                rates.append(r_k)
            acc_kwargs["rate_sch"] = rates

        # Gentle nudge if t wasn’t an integer
        frac = t - round(t)
        if abs(frac) > 1e-12:
            st.info("Note: t is not an integer; the schedule uses n = round(t).")

    elif rate_mode == "Time Varying Continuous Interest":
        expr = st.text_input("Enter δ(x) as a function of x (time)", value="0.04 + 0.01*x")
        try:
            delta_func = make_delta_func(expr)
            acc_kwargs["delta_func"] = delta_func

            # Quick visual of δ(x)
            if t > 0:
                xs = np.linspace(0, float(t), 200)
                ys = [delta_func(xx) for xx in xs]
                fig, ax = plt.subplots()
                ax.plot(xs, ys)
                ax.set_title(f"δ(x) = {expr}")
                ax.set_xlabel("x (time)")
                ax.set_ylabel("δ(x)")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Invalid δ(x): {e}")

    else:  # Continuous Force of Interest (constant δ)
        delta = st.number_input("δ (force of interest)", value=0.04, step=0.001)
        acc_kwargs["del_i"] = delta

    # ——— Compute ———
    if st.button("Compute"):
        pv_input = pv if pv > 0 else None
        fv_input = fv if fv > 0 else None

        # Must provide exactly one of PV or FV
        if (pv_input is None and fv_input is None) or (pv_input is not None and fv_input is not None):
            st.error("Enter exactly one of PV or FV (not both).")
        else:
            if pv_input is None:
                # Solve for PV given FV
                result = pv_lump(fv_input, t, freq=freq, **acc_kwargs)
                st.success(f"PV = {result:,.4f}")
            else:
                # Solve for FV given PV
                result = fv_lump(pv_input, t, freq=freq, **acc_kwargs)
                st.success(f"FV = {result:,.4f}")

# -----------------------------
# Annuity Page
# -----------------------------
if page == "Annuity":
    st.header("Annuity")

    # ---------- Interest / Rate Selection (shared) ----------
    st.subheader("Interest / Rate")
    colr1, colr2 = st.columns([2, 1])
    with colr1:
        rate_mode = st.radio(
            "Rate type",
            [
                "Effective Interest",
                "Simple Interest",
                "Nominal Interest",
                "Continuous Force of Interest",
                "Rate Schedule",
                "Time Varying Continuous Interest",
            ],
            horizontal=True,
            key="ann_rate_mode",
        )
    with colr2:
        # SINGLE freq input (used as payments per year AND as nominal compounding m)
        freq = st.number_input(
            "Payments per year (m)",
            value=1, min_value=1, step=1,
            key="ann_freq"
        )

    acc_kwargs = {}

    if rate_mode == "Effective Interest":
        eff_i = st.number_input("i (effective per year)", value=0.05, step=0.001, key="ann_eff_i")
        acc_kwargs["eff_i"] = eff_i

    elif rate_mode == "Simple Interest":
        simple_i = st.number_input("i (simple per year)", value=0.05, step=0.001, key="ann_simple_i")
        acc_kwargs["simple_i"] = simple_i

    elif rate_mode == "Nominal Interest":
        nom_i = st.number_input("j (nominal per year)", value=0.06, step=0.001, key="ann_nom_i")
        acc_kwargs["nom_i"] = nom_i
        # freq will be used as m when calling functions

    elif rate_mode == "Continuous Force of Interest":
        delta = st.number_input("δ (force of interest per year)", value=0.04, step=0.001, key="ann_delta")
        acc_kwargs["del_i"] = delta

    elif rate_mode == "Rate Schedule":
        st.caption("Enter effective per-period rates (length should cover periods you’ll reference).")
        sch_len = st.number_input("Number of per-period rates to enter", value=12, min_value=1, step=1, key="ann_sch_len")
        rates = []
        cols_per_row = 6
        for k in range(1, sch_len + 1):
            if (k - 1) % cols_per_row == 0:
                row = st.columns(cols_per_row)
            col = row[(k - 1) % cols_per_row]
            with col:
                r_k = st.number_input(
                    f"i[{k}]",
                    value=0.005, step=0.0005, format="%.6f",
                    key=f"ann_rate_sch_{k}",
                )
            rates.append(r_k)
        acc_kwargs["rate_sch"] = rates

    else:  # Time Varying Continuous Interest
        expr = st.text_input("Enter δ(x) function of time x", value="0.04 + 0.01*x", key="ann_delta_expr")
        try:
            dfunc = make_delta_func(expr)         # <-- name it dfunc here
            acc_kwargs["delta_func"] = dfunc
        except Exception as e:
            st.error(f"Invalid δ(x): {e}")
            dfunc = None

    # ---------- Annuity structure ----------
    pay_type = st.selectbox("Payment pattern", ["Level", "Arithmetic", "Geometric"], key="ann_pay_type")
    timing = st.selectbox("Timing", ["Immediate", "Due", "Continuous"], key="ann_timing")
    perpetuity = st.checkbox("Perpetuity (if convergent)", value=False, key="ann_perp")
    years = 0.0
    if not perpetuity:
        years = st.number_input("Term (years)", value=5.0, min_value=0.0, step=1.0, key="ann_years")

    if pay_type == "Level":
        pmt = st.number_input("Payment amount", value=1000.0, step=1.0, min_value=0.0, key="ann_pmt")
        P_0 = step = g = None
    elif pay_type == "Arithmetic":
        P_0 = st.number_input("Starting payment P₀", value=1000.0, step=1.0, min_value=0.0, key="ann_P0")
        step = st.number_input("Step per period (Q)", value=50.0, step=1.0, key="ann_step")
        pmt = None; g = None
    else:  # Geometric
        pmt = st.number_input("First payment P₁", value=1000.0, step=1.0, min_value=0.0, key="ann_geo_pmt")
        g = st.number_input("Growth per period g (decimal)", value=0.02, step=0.001, format="%.4f", key="ann_g")
        P_0 = step = None

    # Map the rate_mode to annuity's i_type & i/kwargs
    if rate_mode == "Effective Interest":
        i_type, i_val, rate_sch = "Effective", eff_i, None
    elif rate_mode == "Simple Interest":
        i_type, i_val, rate_sch = "Simple", simple_i, None
    elif rate_mode == "Nominal Interest":
        i_type, i_val, rate_sch = "Nominal", nom_i, None
    elif rate_mode == "Continuous Force of Interest":
        i_type, i_val, rate_sch = "Force", delta, None
    elif rate_mode == "Rate Schedule":
        i_type, i_val, rate_sch = "Rate Schedule", None, acc_kwargs.get("rate_sch", None)
    else:  # Time Varying Continuous Interest
        i_type, i_val, rate_sch = "Delta Function", None, None  # dfunc set above

    if st.button("Compute Annuity PV / FV", type="primary", key="ann_compute"):
        try:
            PV, FV = annuity(
                pay_type,
                timing=timing,
                perpetuity=perpetuity,
                pmt=pmt,
                years=years,
                freq=int(freq),          # one freq only
                P_0=P_0,
                step=step,
                g=g,
                i_type=i_type,
                i=i_val,
                rate_sch=rate_sch,
                delta_func=acc_kwargs.get("delta_func"),  # pass dfunc if set
            )
            if FV is None:
                st.success(f"PV = {PV:,.4f}")
            else:
                st.success(f"PV = {PV:,.4f}     |     FV @ T={years:g} = {FV:,.4f}")
        except Exception as e:
            st.error(f"Annuity error: {e}")

# -----------------------------
# Portfolio Page
# -----------------------------

if page == "Portfolio":
    CSV_PATH = "portfolio.csv"
    st.header("Portfolio Builder")

    # Initialize session DataFrame
    if "portfolio_df" not in st.session_state:
        try:
            st.session_state.portfolio_df = pd.read_csv(CSV_PATH)
        except FileNotFoundError:
            st.session_state.portfolio_df = pd.DataFrame(
                columns=[
                    "Name", "CashFlowType", "Direction", "Timing",
                    "Perpetuity", "Years", "Freq",
                    "PaymentType", "PaymentParams",
                    "RateType", "RateParams", "OptionParams",
                    "Created"
                ]
            )

    #Input Type
    portfolio_type = st.selectbox("Input Type", ["Structured", "Custom"], key="pb_input_type")
    
    # =========================
    # CUSTOM MODE (CFs and Rates)
    # =========================
    if portfolio_type == "Custom":
        st.subheader("Custom Cash Flows")

        # ---- User Inputs ----

        # Default  Name to Resource (Numbered)
        next_res_num = len(st.session_state.portfolio_df) + 1
        default_name = f"Resource {next_res_num}"

        # Name 
        base_name = st.text_input("Name of Resource", value=default_name, key="pb_name_custom")

        # Years and Freq
        years_input = st.number_input("Years", value=5.0, step=0.5, min_value=0.0, key="pb_custom_years")
        freq_input  = st.number_input("Payments per year (freq)", value=1, step=1, min_value=1, key="pb_custom_freq")

        # Pos/Neg Sign (Mixed will split into pos/neg)
        sign_choice = st.radio(
            "Sign handling for this instrument",
            ["Positive", "Mixed (split into two rows)", "Negative"],
            horizontal=True,
            key="pb_custom_sign_mode"
        )

        # Build/maintain the period grid (freq * time number of rows)
        N = int(np.ceil(float(years_input) * float(freq_input)))
        times = [(k + 1) / float(freq_input) for k in range(max(N, 0))]

        # Initialize or resize the editor DataFrame (preserving prior edits where possible)
        if "custom_cf_df" not in st.session_state:
            st.session_state.custom_cf_df = pd.DataFrame({"Time (years)": [], "Cash Flow": [], "Rate": []})
        df_prev = st.session_state.custom_cf_df.copy()
        st.session_state.custom_cf_df = _resize_editor_df(df_prev, N, times)

        # Editable table (N rows). Leave CF blank = 0; Rate blank will forward-fill on submit.
        st.caption("Fill per-period Cash Flow and Rate. Leave CF blank = 0. Leave Rate blank → uses previous Rate.")
        st.session_state.custom_cf_df = st.data_editor(
            st.session_state.custom_cf_df,
            num_rows="fixed",               # fixed to N rows from Years×Freq
            use_container_width=True,
            key="pb_custom_cf_editor"
        )


        # ---------- Submit ----------
        if st.button("Add to Portfolio", key="pb_add_custom"):
            try:
                # Empty CF -> 0 and Rates are Forward filled
                tbl = _prep_table_for_save(st.session_state.custom_cf_df)

                # Names for Mixed split (Number and Number + 1 for numbered assets)
                m = re.fullmatch(r"\s*Resource\s+(\d+)\s*", base_name)
                base_num = int(m.group(1)) if m else next_res_num
                name_pos = f"Resource {base_num}" if m else f"{base_name} (Pos)"
                name_neg = f"Resource {base_num + 1}" if m else f"{base_name} (Neg)"

                new_rows = []

                # Mixed Split: Same times & rates, split only cash flows by sign
                if sign_choice.startswith("Mixed"):
                    pos_df = tbl.copy()
                    pos_df["Cash Flow"] = np.where(tbl["Cash Flow"] > 0, tbl["Cash Flow"], 0.0)

                    neg_df = tbl.copy()
                    neg_df["Cash Flow"] = np.where(tbl["Cash Flow"] < 0, tbl["Cash Flow"], 0.0)

                    if (pos_df["Cash Flow"].abs().sum() == 0) and (neg_df["Cash Flow"].abs().sum() == 0):
                        st.warning("All cash flows are zero after splitting; nothing to add.")
                    
                    #Add as row to portfolio.csv if legit
                    else:
                        if pos_df["Cash Flow"].abs().sum() > 0:
                            new_rows.append(_row_from_df(name_pos, "Positive", pos_df, years_input, freq_input))
                        if neg_df["Cash Flow"].abs().sum() > 0:
                            new_rows.append(_row_from_df(name_neg, "Negative", neg_df, years_input, freq_input))

                elif sign_choice == "Positive":
                    pos_df = tbl.copy()
                    pos_df["Cash Flow"] = pos_df["Cash Flow"].abs()
                    new_rows.append(_row_from_df(name_pos, "Positive", pos_df, years_input, freq_input))

                else:  # Negative
                    neg_df = tbl.copy()
                    neg_df["Cash Flow"] = - neg_df["Cash Flow"].abs()
                    new_rows.append(_row_from_df(name_pos, "Negative", neg_df, years_input, freq_input))

                if new_rows:
                    st.session_state.portfolio_df = pd.concat(
                        [st.session_state.portfolio_df, pd.DataFrame(new_rows)],
                        ignore_index=True
                    )
                    st.session_state.portfolio_df.to_csv(CSV_PATH, index=False)
                    if len(new_rows) == 2:
                        st.success(f"Saved {new_rows[0]['Name']} and {new_rows[1]['Name']} to {CSV_PATH}")
                    else:
                        st.success(f"Saved {new_rows[0]['Name']} to {CSV_PATH}")

            except ValueError as e:
                st.error(str(e))
    

    # ============================
    # STRUCTURED MODE
    # ============================
    else:
        st.subheader("Structured Cash Flow")
        
        #Defaults
        option_params = {"has_option": False, "type": None, "schedule": [], "price_basis": None}
        perpetuity = False
        
        # ---- User Inputs ----

        # Name
        name = st.text_input("Name of Resource", value=f"Resource {len(st.session_state.portfolio_df)+1}", key="pb_name_struct")

        #CashFlow Type / Sign / Timing of Payment
        c1, c2 = st.columns([1, 1])
        with c1:
            cf_type = st.selectbox("Cash Flow Type", ["Lump Sum", "Annuity", "Annuity + Final Payment"], key="pb_cf_type")
        with c2:
            direction = st.radio("Cash Flow Direction", ["Positive", "Negative"], horizontal=True, key="pb_direction")


        if cf_type == "Lump Sum":
            st.markdown("#### Lump Sum")

            # User Inputs
            pmt = st.number_input("Sum Amount", value=100.0, step=0.01, min_value=0.0, key="sum_amt")
            yr = st.number_input("Year of Payment", value=2.0, step=1.0, min_value=1.0, key="ls_year")  # 1..N
            years = st.number_input("Years (horizon)", value=5.0, step=1.0, min_value=1.0, key="ls_years")

            # Forced/derived settings
            timing = "Immediate"              
            freq = 1
            pay_type = "Custom"             
            
            # Build Payment Parameter
            pay_params = build_lumpsum_payment_params(pmt, yr, years)

            
        elif cf_type == "Annuity + Final Payment":
            st.markdown("#### Annuity + Final Payment")

            # User Input
            pmt = st.number_input("Payment Amount", value=100.0, step=0.01, min_value=0.0, key="pmt_amt")
            fin_pmt = st.number_input("Final Payment", value=100.0, step=0.01, min_value=0.0, key="finpmt_amt")
            years = st.number_input("Years (horizon)", value=5.0, step=1.0, min_value=1.0, key="fin_years")
            freq = st.number_input("Frequency", value=1.0, step=1.0, min_value=1.0, key="fin_freq")
            timing = st.selectbox("Timing", ["Immediate", "Due"], key="pb_timing")

            #Default
            pay_type = "Custom"

            pay_params = build_final_payment_params(pmt, fin_pmt, years, freq, timing)


        else: #Annuity
            timing = st.selectbox("Timing", ["Immediate", "Due", "Continuous"], key="pb_timing")

            #Perpetuity (Bool)
            perpetuity = st.checkbox("Perpetuity", value=False, key="pb_perp")

            # Years / Freq
            d1, d2= st.columns([1, 1])
            with d1:
                years = st.number_input("Years (ignored if perpetuity)", value=5.0, step=1.0, min_value=0.0, key="pb_years")
            with d2:
                freq = st.number_input("Payments per year (freq)", value=1, step=1, min_value=1, key="pb_freq")

            # Payment structure (Level/Arithemtic/Geometric)
            st.markdown("### Payment Structure")
            pay_type = st.selectbox("Payment Structure", ["Level", "Arithmetic", "Geometric"], key="pb_pay_type")
            pay_params = {}

            if pay_type == "Level":
                pay_params["pmt"] = st.number_input("Payment amount", value=100.0, step=1.0, key="pb_pay_level_pmt")
            elif pay_type == "Arithmetic":
                cA, cB = st.columns(2)
                with cA:
                    pay_params["P0"] = st.number_input("Initial payment P0", value=100.0, step=1.0, key="pb_pay_arith_P0")
                with cB:
                    pay_params["step"] = st.number_input("Step increment Q", value=10.0, step=1.0, key="pb_pay_arith_step")
            else: #Geometric
                cA, cB = st.columns(2)
                with cA:
                    pay_params["pmt"] = st.number_input("First payment P1", value=100.0, step=1.0, key="pb_pay_geo_p1")
                with cB:
                    pay_params["g"] = st.number_input("Growth rate g (decimal)", value=0.02, step=0.001, key="pb_pay_geo_g")


        # Rate structure (Effective/Simple/Nominal/Force/Rate Schedule/Delta Function)
        st.markdown("### Rate Structure")
        rate_type = st.selectbox(
            "Rate Type",
            ["Effective", "Simple", "Nominal", "Force", "Rate Schedule", "Delta Function"],
            key="pb_rate_type",
        )
        rate_params = {}
        if rate_type == "Effective":
            rate_params["i"] = st.number_input("Effective i", value=0.05, step=0.001, key="pb_rate_eff_i")
        elif rate_type == "Simple":
            rate_params["i"] = st.number_input("Simple i", value=0.05, step=0.001, key="pb_rate_simple_i")
        elif rate_type == "Nominal":
            cN1, cN2 = st.columns(2)
            with cN1:
                rate_params["j"] = st.number_input("Nominal j", value=0.06, step=0.001, key="pb_rate_nom_j")
            with cN2:
                rate_params["m"] = st.number_input("Compounds per year m", value=2, step=1, key="pb_rate_nom_m")
        elif rate_type == "Force":
            rate_params["delta"] = st.number_input("δ (force of interest)", value=0.04, step=0.001, key="pb_rate_force_delta")
        elif rate_type == "Rate Schedule":
            st.caption("Per-period effective rates. Table length = ceil(Years × Freq). "
                    "Leave blank to use the previous period’s rate. The first rate is required.")

            # periods & time grid based on current years/freq
            N = int(np.ceil(float(years) * float(freq)))
            times_grid = [(k + 1) / float(freq) for k in range(max(N, 0))]

            # key for structured page to distinguish from Custom page
            RS_KEY = "pb_rate_sched_df_struct"
            RS_ED_KEY = "pb_rate_sched_editor_struct"

            # init/resize the editor df to N rows with Time + Rate 
            if RS_KEY not in st.session_state:
                st.session_state[RS_KEY] = pd.DataFrame({"Time (years)": [], "Rate": []})
            st.session_state[RS_KEY] = _resize_editor_df(
                st.session_state[RS_KEY], tgt_len=N, times_grid=times_grid
            )

            # fixed-length editor (N rows). User enters only Rate per period.
            st.session_state[RS_KEY] = st.data_editor(
                st.session_state[RS_KEY],
                num_rows="fixed",
                use_container_width=True,
                key=RS_ED_KEY,
            )

            # Build rate_params on the live state so the Save button picks it up below
            # Forward fill and extract rates(for Cash Flow, give a dummy 0 column)
            rs_tbl = st.session_state[RS_KEY].copy()
            rs_tbl["Cash Flow"] = 0.0 

            try:
                cleaned = _prep_table_for_save(rs_tbl)    
                rate_params["rates"] = _build_rates_from_df(cleaned)
            except ValueError as e:
                # Surface the error early (user can fix before clicking Add to Portfolio)
                st.error(str(e))
                rate_params["rates"] = []
        else:  # Delta Function
            rate_params["expr"] = st.text_input("δ(t) expression", "0.04 + 0.01*t", key="pb_rate_delta_expr")

        # Options schedule (Lump Sum not possible)
        st.markdown("### Optional Early Exercise Schedule")
        disabled_option = (cf_type == "Lump Sum")
        
        if not disabled_option:
            options = st.checkbox("Options?", value=False, key="pb_op")
            if options:
                if "pb_option_sched_df" not in st.session_state:
                    st.session_state.pb_option_sched_df = pd.DataFrame({"Year": [], "PricePer100": []})
                st.session_state.pb_option_sched_df = st.data_editor(
                    st.session_state.pb_option_sched_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="pb_option_sched_editor",
                    disabled=disabled_option,
                )

                option_side = st.selectbox(
                    "Option side (who can exercise)",
                    ["Call", "Put"],
                    help="Call = issuer can repay early; Put = investor can redeem early",
                    key="pb_option_side",
                    disabled=disabled_option,
                )

                # Build schedule list
                schedule = []
                for _, r in st.session_state.pb_option_sched_df.iterrows():
                    y, p = r.get("Year"), r.get("PricePer100")
                    if pd.notnull(y) and pd.notnull(p):
                        try:
                            schedule.append({"year": float(y), "price_per_100": float(p)})
                        except Exception:
                            pass

                # Soft validation
                if schedule and not perpetuity:
                    max_sched_year = max(s["year"] for s in schedule)
                    if max_sched_year >= years:
                        st.warning("One or more exercise years are at/after maturity. Adjust if unintended.")

                use_schedule = (not disabled_option) and len(schedule) > 0
                option_params = {
                    "has_option": bool(use_schedule),
                    "type": (option_side if use_schedule else None),
                    "schedule": (schedule if use_schedule else []),
                    "price_basis": ("per_100_par" if use_schedule else None),
                }

        # Save button
        if st.button("Add to Portfolio", key="pb_add_struct"):
            new_row = {
                "Name": name,
                "CashFlowType": cf_type,
                "Direction": direction,
                "Timing": timing,
                "Perpetuity": perpetuity,
                "Years": years,
                "Freq": int(freq),
                "PaymentType": pay_type,
                "PaymentParams": _jd(pay_params),
                "RateType": rate_type,
                "RateParams": _jd(rate_params),
                "OptionParams": _jd(option_params),
                "Created": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.portfolio_df = pd.concat(
                [st.session_state.portfolio_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.session_state.portfolio_df.to_csv(CSV_PATH, index=False)
            st.success(f"Added {name} to portfolio and saved to {CSV_PATH}")

    # Show current portfolio (preview)
    st.markdown("### Current Portfolio (preview)")
    st.dataframe(st.session_state.portfolio_df, use_container_width=True)

# -----------------------------
# Asset-Liability Management
# -----------------------------
if page == "ALM":
    CSV_PATH = "portfolio.csv"
    st.header("Asset-Liability Management")
    bp = st.number_input("Effective Duration Basis Points", value=25, step=5)

    with st.expander("What do these duration metrics mean?"):
        st.markdown(
            """
            - **Macaulay Duration (D_mac)**: time-weighted average of cash flow times.
            - **Modified Duration (D_mod)**: sensitivity of price to yield:  
            """
            )
        st.latex(r"D_{\text{mod}} = \frac{D_{\text{mac}}}{1 + y/m}")
        st.markdown(
                    """
                        - **DV01**: Dollar change for 1 bp move:  
                    """
                    )
        st.latex(r"\text{DV01} = \frac{\text{PV} \cdot D_{\text{mod}}}{10{,}000}")
        st.markdown(
                    f"""
                    - **Effective Duration (±{bp}bp)**: finite-difference sensitivity using shocked prices:  
                    """
                    )
        st.latex(r"D_{\text{eff}} \approx \frac{P_{\downarrow}-P_{\uparrow}}{2 P_0 \Delta y}, \quad \Delta y = \text{bp}/10{,}000")
        st.markdown(
        """
        - **Duration Gap (A−L)**: **D_mod(Assets)** − **D_mod(Liabilities)** (PV-weighted).
        - **Dollar (Modified) Duration Gap**: $PV_A \\cdot D_A - PV_L \\cdot D_L$ ; target ≈ 0 for immunization.
        """
        )


    if st.button("Compute Durations"):
        res = analyze_portfolio_durations(CSV_PATH, bp)
        st.dataframe(res, use_container_width=True)

        def _clean_numeric(df, cols):
            out = df.copy()
            for c in cols:
                if c not in out:
                    out[c] = np.nan
                out[c] = pd.to_numeric(out[c], errors="coerce")
            return out

        def _agg_side(df_side):
            """Return (PV_total_positive_magnitude, PV_weighted_Dmod, DV01_sum_positive_magnitude, EffDur_weighted)."""
            if df_side.empty:
                return 0.0, 0.0, 0.0, 0.0

            PV_sum = df_side["PV"].sum()

            w = df_side["PV"].abs()
            dmod = (df_side["D_Mod"] * w).sum() / (w.sum() if w.sum() != 0 else 1.0)
            dv01 = (df_side["PV"].abs() * df_side["D_Mod"] / 10_000.0).sum()

            # dynamically use selected bp column if available
            eff_col = f"EffDur_{bp}bp"
            if eff_col in df_side:
                eff = df_side[eff_col]
                eff_w = eff.notna()
                if eff_w.any():
                    eff_dur = (eff[eff_w] * w[eff_w]).sum() / (w[eff_w].sum() if w[eff_w].sum() != 0 else 1.0)
                else:
                    eff_dur = 0.0
            else:
                eff_dur = 0.0

            return float(abs(PV_sum)), float(dmod), float(dv01), float(eff_dur)

        # ensure numeric + derive DV01 per row
        res = _clean_numeric(res, ["PV", "D_Mac", "D_Mod", "PV01"])
        eff_col = f"EffDur_{bp}bp"
        if eff_col not in res:
            st.warning(f"No column '{eff_col}' found — using 0 for Effective Duration.")
            res[eff_col] = np.nan

        res["Row_DV01"] = res["PV"] * res["D_Mod"] / 10_000.0

        assets = res[res["PV"] > 0].copy()
        liabs  = res[res["PV"] < 0].copy()

        PV_A, D_A, DV01_A, Eff_A = _agg_side(assets)
        PV_L, D_L, DV01_L, Eff_L = _agg_side(liabs)

        PV_net = assets["PV"].sum() + liabs["PV"].sum()
        DV01_net = assets["Row_DV01"].sum() + liabs["Row_DV01"].sum()
        Dur_gap = D_A - D_L
        DollarDur_gap = PV_A * D_A - PV_L * D_L

        # ---- Display summary ----
        st.markdown("### Summary by Side")
        colA, colL, colN = st.columns([1,1,1.2])
        with colA:
            st.subheader("Assets")
            st.metric("Total PV", f"{PV_A:,.2f}")
            st.metric("PV-weighted D_mod", f"{D_A:,.4f}")
            st.metric("DV01 (sum)", f"{DV01_A:,.4f}")
            st.metric(f"EffDur (PV-wtd, ±{bp}bp)", f"{Eff_A:,.4f}")
        with colL:
            st.subheader("Liabilities")
            st.metric("Total PV", f"{PV_L:,.2f}")
            st.metric("PV-weighted D_mod", f"{D_L:,.4f}")
            st.metric("DV01 (sum)", f"{DV01_L:,.4f}")
            st.metric(f"EffDur (PV-wtd, ±{bp}bp)", f"{Eff_L:,.4f}")
        with colN:
            st.subheader("Net")
            st.metric("Net PV (A−L)", f"{PV_net:,.2f}")
            st.metric("Duration gap (A−L)", f"{Dur_gap:,.4f}")
            st.metric("Net DV01 (signed)", f"{DV01_net:,.4f}")
            st.metric("Dollar duration gap", f"{DollarDur_gap:,.4f}")

        # ---- Top DV01 contributors ----
        st.markdown("### Largest DV01 Exposure (absolute)")
        top_n = 5
        top_abs = res.assign(Abs_DV01=res["Row_DV01"].abs()).sort_values("Abs_DV01", ascending=False).head(top_n)
        st.dataframe(
            top_abs[["Name", "PV", "D_Mod", "Row_DV01", eff_col]],
            use_container_width=True
        )
