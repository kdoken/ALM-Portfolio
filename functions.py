import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sympy as sp
from typing import Callable, Iterable, Optional
from scipy.integrate import quad
import json

# Accumulation a(t): how $1 grows to time t
def accu_com(
    year: float,                      # time in YEARS (can be fractional)
    freq: int = 1,                    # periods per year (used by nominal & schedules)
    *,
    simple_i: Optional[float] = None, # simple annual rate
    eff_i: Optional[float] = None,    # effective annual rate
    nom_i: Optional[float] = None,    # nominal j with m=freq compounding
    del_i: Optional[float] = None,    # constant force of interest δ
    rate_sch: Optional[Iterable[float]] = None,  # per-period effective rates [i1, i2, ...]
    delta_func: Optional[Callable[[float], float]] = None,  # time-varying force δ(u)
    tol: float = 1e-10
) -> float:
    # sanity checks
    if year < 0:
        raise ValueError("Length of time must be non-negative.")
    modes = [simple_i is not None, eff_i is not None, nom_i is not None,
             del_i is not None, rate_sch is not None, delta_func is not None]
    if sum(modes) != 1:
        raise ValueError("Pick exactly one rate model: simple/eff/nominal/force/schedule/delta_func.")

    # 1) Simple interest: a(t) = 1 + i*t
    if simple_i is not None:
        return 1.0 + simple_i * year

    # 2) Effective annual: a(t) = (1+i)^t
    if eff_i is not None:
        if 1.0 + eff_i <= 0.0:
            raise ValueError("1 + i must be > 0.")
        return (1.0 + eff_i) ** year

    # 3) Nominal j with m=freq: a(t) = (1 + j/m)^(m*t)
    if nom_i is not None:
        if freq <= 0:
            raise ValueError("freq must be positive for nominal rates.")
        step = 1.0 + nom_i / freq
        if step <= 0.0:
            raise ValueError("1 + j/m must be > 0.")
        return step ** (freq * year)

    # 4) Constant force: a(t) = e^{δ t}
    if del_i is not None:
        return math.exp(del_i * year)

    # 5) Per-period schedule: a(n) = ∏(1+i_k), with a fractional last step
    if rate_sch is not None:
        if freq <= 0:
            raise ValueError("freq must be positive for schedules.")
        Nf = year * freq

        # --- fix counting: snap near-integers to exact integer ---
        Nf_round = round(Nf)
        if abs(Nf - Nf_round) <= 1e-9:   # tolerance for float fuzz
            Nf = float(Nf_round)

        n = int(np.floor(Nf))          # full periods
        frac = Nf - n                  # leftover fraction of a period

        prod = 1.0
        it = iter(rate_sch)

        # full periods
        for k in range(n):
            try:
                r = next(it)
            except StopIteration:
                raise ValueError("rate_sch is shorter than needed.")
            if 1.0 + r <= 0.0:
                raise ValueError(f"1 + rate_sch[{k}] must be > 0.")
            prod *= (1.0 + r)

        # fractional last period (use next rate proportionally)
        if frac > 0.0:
            try:
                r = next(it)
            except StopIteration:
                raise ValueError("rate_sch missing rate for fractional period.")
            if 1.0 + r <= 0.0:
                raise ValueError("1 + next schedule rate must be > 0.")
            prod *= (1.0 + r) ** frac

        return prod

    # 6) Time-varying force: a(t) = exp(∫0→t δ(u) du)
    I, _ = quad(delta_func, 0.0, year, epsabs=tol, epsrel=tol)
    return math.exp(I)


# Basic discount and value helpers
def df(t_years: float, *, freq: int = 1, **acc_kwargs) -> float:
    """Discount factor v(t) = 1 / a(t)."""
    return 1.0 / accu_com(t_years, freq=freq, **acc_kwargs)

def pv_lump(fv: float, t_years: float, *, freq: int = 1, **acc_kwargs) -> float:
    """PV of a single future amount."""
    return fv * df(t_years, freq=freq, **acc_kwargs)

def fv_lump(pv: float, t_years: float, *, freq: int = 1, **acc_kwargs) -> float:
    """FV of a single present amount."""
    return pv * accu_com(t_years, freq=freq, **acc_kwargs)

def pv_cashflows(times, cashflows, *, freq: int = 1, **acc_kwargs) -> float:
    """PV of irregular dated cash flows."""
    return float(sum(
        cf * df(float(t), freq=freq, **acc_kwargs)
        for t, cf in zip(times, cashflows)
    ))

def fv_cashflows(times, cashflows, T_years: float, *, freq: int = 1, **acc_kwargs) -> float:
    """FV of irregular dated cash flows to horizon T."""
    return float(sum(
        cf * accu_com(T_years - float(t), freq=freq, **acc_kwargs)
        for t, cf in zip(times, cashflows)
    ))


# Annuity PV & FV  (closed-form when constant per-period; else sum/integral)
def annuity(
    pay_type: str,                     # "Level", "Arithmetic", "Geometric"
    *,
    timing: str,                       # "Immediate", "Due", "Continuous"
    perpetuity: bool = False,          # True for infinite stream (when convergent)
    pmt: Optional[float] = None,       # base payment (also P1 for geometric)
    years: float = 0.0,                # horizon in years (ignored if perpetuity=True)
    freq: int = 1,                     # payments per year
    P_0: Optional[float] = None,       # starting amount for Arithmetic
    step: Optional[float] = None,      # increment per period for Arithmetic (Q)
    g: Optional[float] = None,         # growth per period (decimal) for Geometric
    i_type: Optional[str] = None,      # "Simple", "Effective", "Nominal", "Force", "Rate Schedule", "Delta Function"
    i: Optional[float] = None,         # numeric rate (or list/expr for schedules)
    rate_sch: Optional[Iterable[float]] = None,   # if i_type="Rate Schedule"
    delta_expr: Optional[str] = None,             # if i_type="Delta Function"
    delta_func: Optional[Callable[[float], float]] = None,
    tol: float = 1e-9
):
    # ------------------------
    # Quick sanity checks
    # ------------------------
    valid_timings = {"Immediate", "Due", "Continuous"}
    valid_types   = {"Level", "Arithmetic", "Geometric"}

    if timing not in valid_timings:
        raise ValueError(f"Invalid timing '{timing}'. Must be one of {valid_timings}.")
    if pay_type not in valid_types:
        raise ValueError(f"Invalid pay_type '{pay_type}'. Must be one of {valid_types}.")
    if g is not None and g <= -1:
        raise ValueError("Growth rate g must be greater than -1.")
    if freq <= 0:
        raise ValueError("Payment frequency 'freq' must be positive.")
    if not perpetuity and years <= 0:
        raise ValueError("For finite annuities, 'years' must be > 0.")

    # number of total payments (ignored if perpetuity)
    n = None if perpetuity else int(round(years * freq))
    if n is not None and n <= 0:
        raise ValueError("Number of payments n must be >= 1.")

    # ------------------------
    # Check required payment inputs
    # ------------------------
    if pay_type == "Level":
        if pmt is None or pmt <= 0:
            raise ValueError("Level annuity requires pmt > 0.")
    elif pay_type == "Arithmetic":
        if P_0 is None or step is None:
            raise ValueError("Arithmetic annuity requires P_0 and step.")
    else:  # Geometric
        if pmt is None or pmt <= 0 or g is None:
            raise ValueError("Geometric annuity requires pmt > 0 and g.")

    # ------------------------
    # Build interest model (acc_kwargs)
    # ------------------------
    if i_type is None:
        raise ValueError("Specify i_type for discounting.")
    i_type = i_type.strip()

    acc_kwargs = {}
    constant_per_period = False
    i_per = None

    if i_type == "Simple":
        if i is None:
            raise ValueError("Provide 'i' for Simple rate.")
        acc_kwargs["simple_i"] = float(i)

    elif i_type == "Effective":
        if i is None:
            raise ValueError("Provide 'i' for Effective rate.")
        acc_kwargs["eff_i"] = float(i)
        i_per = (1 + i) ** (1 / freq) - 1
        constant_per_period = True

    elif i_type == "Nominal":
        if i is None:
            raise ValueError("Provide 'i' for Nominal rate.")
        acc_kwargs["nom_i"] = float(i)
        i_per = i / freq
        constant_per_period = True

    elif i_type == "Force":
        if i is None:
            raise ValueError("Provide 'i' for Force (δ).")
        acc_kwargs["del_i"] = float(i)
        i_per = math.exp(i / freq) - 1
        constant_per_period = True

    elif i_type == "Rate Schedule":
        sch = rate_sch if rate_sch is not None else i
        if sch is None:
            raise ValueError("Provide 'rate_sch' or 'i' (list) for Rate Schedule.")
        acc_kwargs["rate_sch"] = list(sch)

    elif i_type == "Delta Function":
        if delta_func is None:
            expr = delta_expr if delta_expr is not None else i
            if expr is None:
                raise ValueError("Provide delta_func or delta_expr/i for Delta Function.")
            acc_kwargs["delta_func"] = make_delta_func(str(expr))
        else:
            acc_kwargs["delta_func"] = delta_func

    else:
        raise ValueError("Unsupported i_type. Must be one of Simple, Effective, Nominal, Force, Rate Schedule, Delta Function.")

    # local helper
    def v(t_years: float) -> float:
        return 1.0 / accu_com(t_years, freq=freq, **acc_kwargs)

    # ------------------------
    # PERPETUITY branch
    # ------------------------
    if perpetuity:
        if timing == "Continuous":
            if "del_i" not in acc_kwargs:
                raise ValueError("Continuous perpetuity requires constant δ (Force).")
            if pay_type != "Level":
                raise ValueError("Continuous perpetuity only supports Level payments.")
            return pmt / float(i), None  # simple closed form

        # Discrete perpetuity
        if not constant_per_period:
            # fallback to convergence tail sum
            max_k = 10000
            pv, k = 0.0, 0
            while k < max_k:
                t = k / freq
                if pay_type == "Level":
                    Pk = pmt
                elif pay_type == "Arithmetic":
                    Pk = P_0 + k * step
                else:
                    Pk = pmt * (1 + g) ** k
                term = Pk * v(t)
                pv += term
                if abs(term) < tol * max(1.0, abs(pv)):
                    break
                k += 1
            return pv, None

        # constant per-period perpetuity closed forms
        if i_per <= 0:
            raise ValueError("Perpetuity requires positive per-period rate.")
        mult = (1 + i_per) if timing == "Due" else 1.0
        if pay_type == "Level":
            return (pmt / i_per) * mult, None
        if pay_type == "Arithmetic":
            return ((P_0 / i_per) + (step / i_per**2)) * mult, None
        if pay_type == "Geometric":
            if i_per <= g:
                raise ValueError("Geometric perpetuity requires i_per > g.")
            return (pmt / (i_per - g)) * mult, None

    # ------------------------
    # FINITE annuity — closed form when rate constant
    # ------------------------
    if timing in ("Immediate", "Due") and constant_per_period:
        i_ = i_per
        if i_ <= 0:
            raise ValueError("Per-period rate must be > 0 for closed-form.")
        v_ = 1 / (1 + i_)
        v_n = v_ ** n
        a_n = (1 - v_n) / i_
        s_n = ((1 + i_) ** n - 1) / i_

        if pay_type == "Level":
            pv, fv = pmt * a_n, pmt * s_n
        elif pay_type == "Arithmetic":
            pv = (P_0 * a_n) + (step * (a_n - n * v_n) / i_)
            fv = pv * (1 + i_) ** n
        else:  # Geometric
            if i_ <= g:
                raise ValueError("For geometric annuity, need i_per > g.")
            pv = pmt * (1 - ((1 + g) / (1 + i_)) ** n) / (i_ - g)
            fv = pv * (1 + i_) ** n

        if timing == "Due":
            pv *= (1 + i_)
            fv *= (1 + i_)
        return pv, fv

    # ------------------------
    # FINITE annuity — variable rate or continuous
    # ------------------------
    if timing == "Continuous":
        if pay_type != "Level":
            raise ValueError("Continuous annuity currently supports Level payments only.")
        PV, _ = quad(lambda t: pmt * v(t), 0.0, years, epsabs=tol, epsrel=tol)
        FV, _ = quad(lambda t: pmt * accu_com(years - t, freq=freq, **acc_kwargs), 0.0, years, epsabs=tol, epsrel=tol)
        return PV, FV

    # general sum (variable rates)
    times = np.array(
        [k / freq for k in range(1, n + 1)] if timing == "Immediate"
        else [k / freq for k in range(0, n)],
        dtype=float
    )
    if pay_type == "Level":
        pays = np.full(times.shape, float(pmt))
    elif pay_type == "Arithmetic":
        ks = np.arange(1 if timing == "Immediate" else 0, len(times) + (1 if timing == "Immediate" else 0))
        pays = P_0 + (ks - (1 if timing == "Immediate" else 0)) * step
    else:
        ks = np.arange(1 if timing == "Immediate" else 0, len(times) + (1 if timing == "Immediate" else 0))
        pays = pmt * (1 + g) ** (ks - (1 if timing == "Immediate" else 0))

    dfs = np.array([v(t) for t in times])
    PV = float(np.sum(pays * dfs))
    growth = np.array([accu_com(years - t, freq=freq, **acc_kwargs) for t in times])
    FV = float(np.sum(pays * growth))
    return PV, FV



# make_delta_func(expr): Build a callable δ(u) from a math expression string
def make_delta_func(expr: str):
    """
    Convert a string like "0.04 + 0.01*x" into a callable delta(u) → float.
    Allowed symbols: x, pi, e, and common math functions.
    """
    # define symbol + safe math namespace
    t = sp.symbols('t', real=True)
    allowed = {
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'pi': sp.pi, 'e': sp.E, 't': t
    }

    # parse safely
    try:
        sym = sp.sympify(expr, locals=allowed)
    except sp.SympifyError as e:
        raise ValueError(f"Could not parse expression: {e}")

    # turn symbolic to numpy-based callable
    f = sp.lambdify(t, sym, modules=[
        {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh
        },
        'numpy'
    ])

    # wrapper: evaluate and ensure scalar finite output
    def delta(u: float) -> float:
        val = f(u)
        try:
            val = float(val)
        except Exception:
            val = float(np.asarray(val).reshape(()))
        if not math.isfinite(val):
            raise ValueError(f"δ({u}) is not finite.")
        return val

    return delta



#Utility Functions
def _j(s):
    """Load JSON from CSV cells that may be double-quoted. Fall back gracefully."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    if isinstance(s, (dict, list)):   # already parsed
        return s
    try:
        return json.loads(s)
    except Exception:
        # common: CSV escapes "" → turn them into "
        try:
            return json.loads(str(s).replace('""', '"'))
        except Exception:
            return {}

def _sgn(direction):  # "Positive"/"Negative"
    return 1.0 if str(direction).lower().startswith("pos") else -1.0

def _safe_int(x, default=1): # Safely convert any value to an integer. Returns a default value if x is None, NaN, or invalid.
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return int(round(float(x)))
    except Exception:
        return default

def _safe_float(x, default=0.0): # Safely convert any value to an float. Returns a default value if x is None, NaN, or invalid.
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default

# ---------- rate plumbing for accu_com ----------
def build_acc_kwargs(rate_type: str, rate_params: dict, freq: int):
    rt = (rate_type or "").strip()
    rp = rate_params or {}
    kw = {}

    if rt == "Effective":
        kw["eff_i"] = _safe_float(rp.get("i"), 0.0)
    elif rt == "Simple":
        kw["simple_i"] = _safe_float(rp.get("i"), 0.0)
    elif rt == "Nominal":
        kw["nom_i"] = _safe_float(rp.get("j"), 0.0)
    elif rt == "Force":
        kw["del_i"] = _safe_float(rp.get("delta"), 0.0)
    elif rt == "Rate Schedule":
        # rp can be {"rates":[...]} OR directly a list [...]
        if isinstance(rp, list):
            rates = rp
        else:
            rates = rp.get("rates", [])
        if isinstance(rates, str):
            try:
                rates = json.loads(rates)
            except Exception:
                rates = [float(x.strip()) for x in rates.split(",") if x.strip() != ""]
        kw["rate_sch"] = [float(r) for r in rates]
    elif rt == "Delta Function":
        # rpars may be a JSON string → dict
        if isinstance(rp, str):
            try:
                rp = json.loads(rp)
            except Exception:
                rp = {}
        expr = rp.get("expr", None) if isinstance(rp, dict) else None
        if expr is None:
            raise ValueError("Delta Function requires 'expr'.")
        kw["delta_func"] = make_delta_func(str(expr))
    else:
        raise ValueError(f"Unsupported RateType '{rate_type}'.")
    return kw

def per_period_rate_if_constant(acc_kwargs: dict, freq: int):
    """
    If the discount model implies a constant per-period effective rate (for closed-form),
    return i_per; else return None.
    """
    if "eff_i" in acc_kwargs:
        return (1.0 + acc_kwargs["eff_i"])**(1.0/freq) - 1.0
    if "nom_i" in acc_kwargs:
        return acc_kwargs["nom_i"] / float(freq)
    if "del_i" in acc_kwargs:
        return math.exp(acc_kwargs["del_i"]/float(freq)) - 1.0
    # simple_i, rate_sch, delta_func are not constant per period
    return None

# cash-flow builder from a single row 
def row_to_cashflows(row: pd.Series):
    cf_type    = str(row.get("CashFlowType","Lump Sum"))
    timing     = str(row.get("Timing","Immediate"))
    perpetuity = bool(row.get("Perpetuity", False))
    years      = max(0.0, _safe_float(row.get("Years"), 0.0))
    freq       = max(1,   _safe_int(row.get("Freq"), 1))
    pay_type   = str(row.get("PaymentType","Level"))
    pay        = _j(row.get("PaymentParams"))
    sgn        = _sgn(row.get("Direction","Positive"))

    if perpetuity:
        return None

    cfs = []

    # Treat these three types similarly when PaymentParams is a list of rows
    if cf_type in ("Custom", "Annuity + Final Payment"):
        if isinstance(pay, list):
            for r in pay:
                # tolerate either editor keys or short keys
                t = _safe_float(r.get("Time (years)", r.get("t", 0.0)), 0.0)
                a = _safe_float(r.get("Cash Flow",  r.get("cf", 0.0)), 0.0)
                cfs.append((t, sgn * a))
        elif isinstance(pay, dict) and "times" in pay and "amounts" in pay:
            cfs = [(float(t), sgn*float(a)) for t, a in zip(pay["times"], pay["amounts"])]

        if cfs:
            cfs.sort(key=lambda x: x[0])
            merged = []
            for t, a in cfs:
                if merged and abs(t - merged[-1][0]) < 1e-12:
                    merged[-1] = (merged[-1][0], merged[-1][1] + a)
                else:
                    merged.append((t, a))
            return merged
        return []

    if cf_type == "Lump Sum":
        # Support BOTH dict {pmt:...} and list-of-rows styles
        if isinstance(pay, dict) and "pmt" in pay:
            amt = _safe_float(pay.get("pmt"), 0.0)
            t = years if years > 0 else 0.0
            return [(t, sgn * amt)]
        elif isinstance(pay, list):
            # interpret like Custom: use provided rows
            for r in pay:
                t = _safe_float(r.get("Time (years)", r.get("t", 0.0)), 0.0)
                a = _safe_float(r.get("Cash Flow",  r.get("cf", 0.0)), 0.0)
                cfs.append((t, sgn * a))
            if cfs:
                cfs.sort(key=lambda x: x[0])
                merged = []
                for t, a in cfs:
                    if merged and abs(t - merged[-1][0]) < 1e-12:
                        merged[-1] = (merged[-1][0], merged[-1][1] + a)
                    else:
                        merged.append((t, a))
                return merged
            return []
        else:
            return []  # nothing usable

    if cf_type == "Annuity":
        n = _safe_int(years * freq, 0)
        if n <= 0:
            return []
        if timing == "Immediate":
            times = [k/freq for k in range(1, n+1)]; shift = 1
        elif timing == "Due":
            times = [k/freq for k in range(0, n)];   shift = 0
        else:  # "Continuous" → approximate on discrete grid
            times = [k/freq for k in range(1, n+1)]; shift = 1

        if pay_type == "Level":
            pmt = _safe_float(pay.get("pmt"), 0.0) if isinstance(pay, dict) else 0.0
            return [(t, sgn * pmt) for t in times]

        if pay_type == "Arithmetic":
            P0   = _safe_float(pay.get("P0"), 0.0)   if isinstance(pay, dict) else 0.0
            step = _safe_float(pay.get("step"), 0.0) if isinstance(pay, dict) else 0.0
            pays = [P0 + (k - (1 if timing=="Immediate" else 0))*step
                    for k, _ in enumerate(times, start=shift)]
            return [(t, sgn * p) for t, p in zip(times, pays)]

        if pay_type == "Geometric":
            P1 = _safe_float(pay.get("pmt"), 0.0) if isinstance(pay, dict) else 0.0
            g  = _safe_float(pay.get("g"), 0.0)   if isinstance(pay, dict) else 0.0
            if g <= -1:
                return []
            pays = [P1 * (1+g)**(k - (1 if timing=="Immediate" else 0))
                    for k, _ in enumerate(times, start=shift)]
            return [(t, sgn * p) for t, p in zip(times, pays)]

        return []

    # leave other types harmless
    return []

# ---------- PV & durations over finite CFs ----------
def _df_from_kwargs(t, freq=1, **acc_kwargs):
    return 1.0 / accu_com(float(t), freq=freq, **acc_kwargs)

def pv_irregular(cashflows, freq=1, **acc_kwargs):
    return float(sum(float(cf) * _df_from_kwargs(float(t), freq=freq, **acc_kwargs)
                     for t, cf in cashflows))

def duration_metrics(cashflows, freq=1, **acc_kwargs):
    """
    Macaulay/Modified duration for a *finite* CF set.
    - Macaulay uses time-weighted PV / PV
    - Modified divides by (1 + effective annual i) inferred from DF(1)
    """
    if not cashflows:
        return {"PV": 0.0, "D_mac": 0.0, "D_mod": 0.0, "PV01": 0.0}

    PV = pv_irregular(cashflows, freq=freq, **acc_kwargs)
    if abs(PV) < 1e-15:
        return {"PV": 0.0, "D_mac": 0.0, "D_mod": 0.0, "PV01": 0.0}

    numer = sum(float(t) * float(cf) * _df_from_kwargs(float(t), freq=freq, **acc_kwargs)
                for t, cf in cashflows)
    D_mac = numer / PV

    # Modified duration via one-year effective rate implied by df(1).
    i_eff_1y = (1.0 / _df_from_kwargs(1.0, freq=freq, **acc_kwargs)) - 1.0
    D_mod = D_mac / (1.0 + i_eff_1y)
    PV01 = D_mod * PV / 10_000.0
    return {"PV": PV, "D_mac": D_mac, "D_mod": D_mod, "PV01": PV01}


def _shock_kwargs(kw, d):
    kw = dict(kw)
    if "eff_i" in kw: kw["eff_i"] += d
    elif "nom_i" in kw: kw["nom_i"] += d
    elif "simple_i" in kw: kw["simple_i"] += d
    elif "del_i" in kw: kw["del_i"] += d
    elif "rate_sch" in kw: kw["rate_sch"] = [r + d for r in kw["rate_sch"]]
    elif "delta_func" in kw:
        f = kw["delta_func"]
        kw["delta_func"] = (lambda g=f, add=d: (lambda x: float(g(x)) + add))()
    return kw

def effective_duration(cashflows, freq=1, delta_bp=25.0, **acc_kwargs):
    """
    Bump-and-reprice duration (parallel bump of the *model you passed*).
    Works with any discount setup because we just reprice cashflows.
    """
    if not cashflows:
        return 0.0
    P0 = pv_irregular(cashflows, freq=freq, **acc_kwargs)
    if abs(P0) < 1e-15:
        return 0.0

    d = delta_bp / 10_000.0
    Pp = pv_irregular(cashflows, freq=freq, **_shock_kwargs(acc_kwargs, +d))
    Pm = pv_irregular(cashflows, freq=freq, **_shock_kwargs(acc_kwargs, -d))
    return (Pm - Pp) / (2.0 * P0 * d)


def duration_perpetuity(pay_type, timing, freq, pay_params, **acc_kwargs):
    """
    Perpetuity duration when a constant per-period rate exists.
    Level / Arithmetic / Geometric; timing Immediate/Due.
    """
    i_per = per_period_rate_if_constant(acc_kwargs, freq)
    if i_per is None or i_per <= 0:
        raise ValueError("Perpetuity needs a constant positive per-period rate.")

    timing_mult = (1.0 + i_per) if timing == "Due" else 1.0

    if pay_type == "Level":
        c = _safe_float(pay_params.get("pmt"), 0.0)
        PV = (c / i_per) * timing_mult
        D_mac = (1.0 + i_per) / i_per
        D_mod = D_mac / (1.0 + i_per)
        PV01 = D_mod * PV / 10_000.0
        return {"PV": PV, "D_mac": D_mac, "D_mod": D_mod, "PV01": PV01}

    if pay_type == "Arithmetic":
        P0   = _safe_float(pay_params.get("P0"), 0.0)
        step = _safe_float(pay_params.get("step"), 0.0)
        PV = ((P0 / i_per) + (step / (i_per**2))) * timing_mult
        # Estimate duration numerically with a long tail (robust and simple)
        N = 10_000
        times = [k/freq for k in range(1, N+1)]
        pays  = [P0 + (k-1)*step for k in range(1, N+1)]
        dm = duration_metrics(list(zip(times, pays)), freq=freq, **acc_kwargs)
        if timing == "Due":
            # shift one period forward by scaling payments; ratio cancels cleanly
            dm = duration_metrics([(t, c*(1.0+i_per)) for (t,c) in zip(times, pays)], freq=freq, **acc_kwargs)
        return {"PV": PV, "D_mac": dm["D_mac"], "D_mod": dm["D_mod"], "PV01": dm["PV01"]}

    if pay_type == "Geometric":
        P1 = _safe_float(pay_params.get("pmt"), 0.0)
        g  = _safe_float(pay_params.get("g"), 0.0)
        if g <= -1:
            raise ValueError("Growth g must be > -1.")
        if i_per <= g:
            raise ValueError("Geometric perpetuity needs i_per > g.")
        PV = (P1 / (i_per - g)) * timing_mult
        N = 10_000
        times = [k/freq for k in range(1, N+1)]
        pays  = [P1 * (1+g)**(k-1) for k in range(1, N+1)]
        dm = duration_metrics(list(zip(times, pays)), freq=freq, **acc_kwargs)
        if timing == "Due":
            dm = duration_metrics([(t, c*(1.0+i_per)) for (t,c) in zip(times, pays)], freq=freq, **acc_kwargs)
        return {"PV": PV, "D_mac": dm["D_mac"], "D_mod": dm["D_mod"], "PV01": dm["PV01"]}

    raise ValueError("Perpetuity pay_type must be Level, Arithmetic, or Geometric.")


def analyze_portfolio_durations(csv_path: str, bp) -> pd.DataFrame:
    """
    Read portfolio.csv → per-row PV, Macaulay/Modified duration, PV01, EffDur(25bp).
    """
    df = pd.read_csv(csv_path)
    out_rows = []

    for idx, row in df.iterrows():
        name = row.get("Name", f"Row{idx+1}")
        notes = []
        try:
            # interest model
            freq  = max(1, _safe_int(row.get("Freq"), 1))
            rtype = str(row.get("RateType","Effective"))
            rpars = _j(row.get("RateParams"))
            acc_kwargs = build_acc_kwargs(rtype, rpars, freq=freq)

            perpetuity = bool(row.get("Perpetuity", False))
            pay_type   = str(row.get("PaymentType","Level"))
            timing     = str(row.get("Timing","Immediate"))
            pay_params = _j(row.get("PaymentParams"))

            if perpetuity:
                metrics = duration_perpetuity(pay_type, timing, freq, pay_params, **acc_kwargs)
                i_per = per_period_rate_if_constant(acc_kwargs, freq=freq)
                if i_per is not None:
                    eff_dur = float(metrics["D_mod"])
                else:
                    eff_dur = np.nan
                    notes.append("Perpetuity under non-constant rate model: EffDur left NaN.")
            else:
                cfs = row_to_cashflows(row)
                if cfs is None:
                    notes.append("Perpetuity flagged but no closed-form path taken.")
                    raise ValueError("Perpetuity handling mismatch.")
                metrics = duration_metrics(cfs, freq=freq, **acc_kwargs)
                eff_dur = effective_duration(cfs, freq=freq, delta_bp=25.0, **acc_kwargs)

            out_rows.append({
                "Name": name,
                "PV": metrics["PV"],
                "D_Mac": metrics["D_mac"],
                "D_Mod": metrics["D_mod"],
                "PV01": metrics["PV01"],
                f"EffDur_{bp}bp": (float(eff_dur) if not pd.isna(eff_dur) else np.nan),
                "Notes": "; ".join(notes) if notes else ""
            })

        except Exception as e:
            out_rows.append({
                "Name": name,
                "PV": np.nan, "D_Mac": np.nan, "D_Mod": np.nan, "PV01": np.nan,
                f"EffDur_{bp}bp": np.nan,
                "Notes": f"Error: {e}"
            })

    return pd.DataFrame(out_rows)