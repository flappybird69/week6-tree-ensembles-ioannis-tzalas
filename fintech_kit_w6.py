from __future__ import annotations
import logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

START             = pd.Timestamp("2014-09-17")                       # BTC's first Yahoo date — the binding asset since Week 1
TEST_START        = pd.Timestamp("2024-10-01")                       # Week 4's sealed holdout. Declared here, before any model exists.
DEFAULT_TICKERS   = {"spy": "SPY", "eurusd": "EURUSD=X", "btc": "BTC-USD"}
DEFAULT_CLOSE_UTC = {"spy": 21, "eurusd": 22, "btc": 24}             # daily-bar close, hours UTC: the clock rule the as-of test cannot see
PPY               = {"spy": 252, "eurusd": 252, "btc": 365}          # periods per year, per calendar (add your assets)
_END_SYN          = pd.Timestamp("2026-10-23")

def download_prices(tickers: dict = DEFAULT_TICKERS, start=START, data_dir="data") -> None:
    """Refresh data/{stem}.csv via yfinance, one file per ticker on ITS OWN calendar. Silent failure → files or fallback are used."""
    Path(data_dir).mkdir(exist_ok=True)
    try:
        import yfinance as yf
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        raw = yf.download(list(tickers.values()), start=start, auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if close.dropna(how="all").empty:
            raise RuntimeError("empty download")
        for stem, sym in tickers.items():
            s = close[sym].dropna() if sym in close.columns else pd.Series(dtype=float)
            if len(s) < 500:
                print(f"⚠️  {stem}: only {len(s)} rows, not saved"); continue
            s.rename("Close").to_frame().to_csv(f"{data_dir}/{stem}.csv"); print(f"✅ {stem}: {len(s):,} rows saved")
    except Exception as e:
        print(f"⚠️  Download unavailable ({type(e).__name__}). Will use existing files in {data_dir}/ or the synthetic fallback.")

def _load_price_file(stem: str, data_dir: str) -> pd.Series | None:
    for ext in (".csv", ".xlsx"):
        f = Path(data_dir) / f"{stem}{ext}"
        if not f.exists():
            continue
        df = pd.read_csv(f) if ext == ".csv" else pd.read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower().startswith("date")), df.columns[0])
        px_col   = next((c for c in df.columns if c.lower() in ("close", "adj close", "adj_close")), None)
        if px_col is None:
            return None
        s = pd.Series(pd.to_numeric(df[px_col], errors="coerce").values, index=pd.to_datetime(df[date_col], errors="coerce"))
        s = s[s.index.notna()].sort_index().dropna()
        return s if len(s) > 500 else None
    return None

def _arma_garch_path(rng, n, phi, theta, mu, sig, alpha, beta, df, phi_calm=None):
    """Week 4's generator + the Week 6 twist: the lag-1 coefficient is phi_calm on CALM days (today's conditional variance,
       known at yesterday's close, at or below its long-run level) and phi on stormy ones — an interaction, not a line."""
    omega = sig**2 * (1 - alpha - beta)
    z = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    r, e, s2 = np.zeros(n), np.zeros(n), np.full(n, sig**2)
    for t in range(n):
        if t:
            s2[t] = omega + alpha * e[t-1]**2 + beta * s2[t-1]
        e[t] = np.sqrt(s2[t]) * z[t]
        ar = phi_calm if (phi_calm is not None and s2[t] <= sig**2) else phi
        r[t] = mu + e[t] + (ar * (r[t-1] - mu) + theta * e[t-1] if t else 0.0)
    return r

_TRUE = {  # calendar,  S0,   phi (storm), theta,  mu/day,    sig/day,            alpha, beta, df, stream, phi_calm
    "spy":    ("B", 200.0,  0.00, -0.08,  0.08/252,  0.17/np.sqrt(252), 0.09, 0.88, 6,  3,  None),
    "eurusd": ("B",   1.29, 0.00,  0.00,  0.00,      0.08/np.sqrt(252), 0.04, 0.94, 8,  6,  None),
    "btc":    ("D", 450.0,  0.00,  0.00,  0.30/365,  0.60/np.sqrt(365), 0.10, 0.86, 4, 10,  0.25),   # momentum ONLY on calm days
}

def _synthetic_market() -> dict:
    prices = {}
    for stem, (cal, s0, phi, theta, mu, sig, a, b, df, stream, phi_calm) in _TRUE.items():
        rng = np.random.default_rng([42, stream])
        idx = pd.date_range(START, _END_SYN, freq=cal)
        r = _arma_garch_path(rng, len(idx), phi, theta, mu, sig, a, b, df, phi_calm)
        prices[stem] = pd.Series(s0 * np.exp(np.cumsum(r)), index=idx)
    return prices

def load_prices(tickers: dict = DEFAULT_TICKERS, data_dir="data") -> tuple[dict, bool]:
    """Close series per stem from data/. If any of the default trio is missing, the WHOLE trio is replaced by the synthetic market
       (returns synthetic=True). Any other stem must have a file — the kit cannot invent your assets."""
    loaded = {stem: _load_price_file(stem, data_dir) for stem in tickers}
    missing_trio  = [s for s in DEFAULT_TICKERS if s in tickers and loaded[s] is None]
    missing_other = [s for s in tickers if s not in DEFAULT_TICKERS and loaded[s] is None]
    if missing_other:
        raise FileNotFoundError(f"no usable price file for {missing_other} in {data_dir}/ — run download_prices() or add data/<stem>.csv with a Date and a Close column")
    synthetic = bool(missing_trio)
    if synthetic:
        loaded.update(_synthetic_market())
        print(f"⚠️  SYNTHETIC fallback ({', '.join(missing_trio)} missing): Week 4's ARMA–GARCH market, with BTC's calm-day momentum.")
    else:
        print("✅ REAL DATA: all assets loaded from data/")
    for stem in tickers:
        s = loaded[stem]; print(f"   {stem:7s} {len(s):>5,} rows | {s.index[0].date()} → {s.index[-1].date()}")
    print(f"🔒 Holdout sealed: everything from {TEST_START.date()} is test data. No rung and no model is chosen on it.")
    return loaded, synthetic

def log_returns(prices: dict) -> dict:
    """Daily log returns in PERCENT, one series per stem, each on its own calendar."""
    return {stem: (100 * np.log(prices[stem]).diff().dropna()).rename(stem) for stem in prices}

def feature_factory(stem: str, ret: dict, prices: dict, close_utc: dict = DEFAULT_CLOSE_UTC) -> pd.DataFrame:
    """~48 predictors + Week 5's three audited agent features — every one computable at the close of day t.
       Cross-asset features use the LAST KNOWN value of every other stem in `ret` (as-of merge), lagged once more when that
       stem's bar closes after ours (the clock rule). Every stem needs an entry in close_utc."""
    missing = [s for s in ret if s not in close_utc]
    if missing:
        raise KeyError(f"close_utc has no entry for {missing}: state when each bar closes (hours UTC) before using cross-asset features")
    r = ret[stem]; lp = np.log(prices[stem]).reindex(r.index)
    F = pd.DataFrame(index=r.index)
    for k in range(1, 11):
        F[f"r_lag{k}"] = r.shift(k - 1)
    for k in range(1, 6):
        F[f"abs_lag{k}"] = r.abs().shift(k - 1)
        F[f"sq_lag{k}"]  = (r**2).shift(k - 1)
    for w in (5, 21, 63):
        F[f"mom{w}"] = r.rolling(w).mean()
        F[f"vol{w}"] = r.rolling(w).std()
    F["vol126"]    = r.rolling(126).std()
    F["rv_week"]   = (r**2).rolling(5).mean()
    F["rv_month"]  = (r**2).rolling(21).mean()
    F["vol_ratio"] = F["vol5"] / F["vol63"]
    F["z21"]       = r / F["vol21"]
    F["dd63"]      = lp - lp.rolling(63).max()
    F["mr21"]      = lp - lp.rolling(21).mean()
    F["sign_lag1"] = np.sign(r)
    runs = (np.sign(r) != np.sign(r).shift()).cumsum()
    F["streak"]    = np.sign(r) * (np.sign(r).groupby(runs).cumcount() + 1)
    dow = r.index.dayofweek
    for d, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
        if d: F[f"dow_{name}"] = (dow == d).astype(float)
    for other in ret:
        if other == stem: continue
        o = ret[other]
        if close_utc[other] >= close_utc[stem]:
            o = o.shift(1)
        F[f"{other}_lag1"]  = o.reindex(r.index, method="ffill")
        F[f"{other}_mom5"]  = o.rolling(5).mean().reindex(r.index, method="ffill")
        F[f"{other}_vol21"] = o.rolling(21).std().reindex(r.index, method="ffill")
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    F["rsi14"]    = up.rolling(14).mean() / (up.rolling(14).mean() + dn.rolling(14).mean())
    F["sharpe21"] = r.rolling(21).mean() / r.rolling(21).std()
    ref = "spy" if (stem != "spy" and "spy" in ret) else next((s for s in ret if s != stem), stem)
    F["rel_vs_spy"] = r - ret[ref].reindex(r.index, method="ffill")
    return F.loc[:, F.nunique() > 1]

def make_dataset(stem: str, ret: dict, prices: dict, close_utc: dict = DEFAULT_CLOSE_UTC):
    """Rows indexed by the TARGET date: row s = the return r_s and the features known at the PREVIOUS close. Returns (X, y)."""
    X = feature_factory(stem, ret, prices, close_utc).shift(1)
    y = ret[stem].rename("r")
    keep = X.dropna().index.intersection(y.dropna().index)
    return X.loc[keep], y.loc[keep]

def as_of_audit(stem: str, ret: dict, prices: dict, close_utc: dict = DEFAULT_CLOSE_UTC, n_dates: int = 8, seed: int = 42) -> dict:
    """THE LEAK DETECTOR (rule T1). Rebuild the features on a sample truncated at date t; anything already computed must not change.
       Returns {offending column: diagnosis}; an empty dict is a pass. Pass extra=... through a wrapper if you add features."""
    rng = np.random.default_rng(seed)
    full = feature_factory(stem, ret, prices, close_utc)
    dates = rng.choice(full.index[300:-2], n_dates, replace=False)
    offenders = {}
    for t in sorted(dates):
        cut = feature_factory(stem, {s: ret[s].loc[:t] for s in ret}, {s: prices[s].loc[:t] for s in prices}, close_utc)
        ref = full.loc[cut.index, cut.columns]
        for col in cut.columns:
            if not np.allclose(cut[col].values, ref[col].values, equal_nan=True):
                offenders[col] = "forward-looking window" if (cut[col].isna() & ref[col].notna()).any() else "full-sample statistic"
    return offenders

def r2_os(y, yhat, bench) -> float:
    """Campbell–Thompson out-of-sample R²: 1 − MSE(model)/MSE(benchmark). The benchmark is the TRAINING mean, never the test mean."""
    return float(1 - ((y - yhat)**2).sum() / ((y - bench)**2).sum())

def pipe(model):
    """Standardize, then fit — ONE object, so the scaler is fitted inside the fold. For LINEAR models only; trees need no scaling."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])

def cv_splitter(n_splits: int = 5):
    """The ONE place the fold design lives: expanding windows in time order. Your eval suite should assert on its type."""
    return TimeSeriesSplit(n_splits=n_splits)

def ts_cv(model_fn, F, y, n_splits: int = 5) -> np.ndarray:
    """Time-ordered CV: per-fold R²_OS against the TRAINING-fold mean. model_fn is a zero-argument factory returning an unfitted model."""
    scores = []
    for tr, va in cv_splitter(n_splits).split(F):
        m = model_fn().fit(F.iloc[tr], y.iloc[tr])
        scores.append(r2_os(y.iloc[va], m.predict(F.iloc[va]), y.iloc[tr].mean()))
    return np.array(scores)

def summarize(scores) -> tuple[float, float]:
    """Per-fold scores → (mean, standard error)."""
    return float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

def one_se_pick(means, ses) -> int:
    """Index of the SIMPLEST rung (lowest index) whose mean score is within one se of the best rung's. Order your rungs simplest-first."""
    means, ses = np.asarray(means, float), np.asarray(ses, float); best = int(np.argmax(means))
    return int(np.argmax(means >= means[best] - ses[best]))

def rolling_forecast(F, y, model_fn, test_start=TEST_START, refit: int = 63):
    """For every target date s ≥ test_start: the forecast of y_s from a model fitted on rows BEFORE the last refit date, frozen between
       refits (Week 4's cadence). Returns (forecasts, the model fitted at the first refit date)."""
    idx = F.index; t0 = int(np.searchsorted(idx, test_start)); fc = pd.Series(np.nan, index=idx[t0:]); first = None
    for start in range(t0, len(idx), refit):
        stop = min(start + refit, len(idx))
        m = model_fn().fit(F.iloc[:start], y.iloc[:start])
        if first is None: first = m
        fc.iloc[start - t0: stop - t0] = m.predict(F.iloc[start:stop])
    return fc, first

def dm_test(loss1: pd.Series, loss2: pd.Series) -> tuple[float, float]:
    """Diebold–Mariano: H0 E[loss1 − loss2] = 0; HAC t-statistic on the differential. NEGATIVE t favours model 1. Returns (t, p)."""
    d = (loss1 - loss2).dropna(); T = len(d)
    q = int(np.floor(4 * (T / 100) ** (2 / 9)))
    ols = sm.OLS(d.values, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": q})
    return float(ols.tvalues[0]), float(ols.pvalues[0])

def verdict(t: float, p: float, rival: str) -> str:
    """The three-way reading of a DM statistic."""
    return (f"beats {rival} beyond noise ✅" if (p < .05 and t < 0) else f"LOSES to {rival} beyond noise ❌" if (p < .05 and t > 0)
            else f"ties {rival}: no evidence either way")

def qlike(var_fc: pd.Series, r2: pd.Series) -> pd.Series:
    """Patton (2011) loss for variance forecasts against the squared-return proxy; robust to the proxy's noise."""
    x = np.maximum(r2, 1e-10) / var_fc
    return x - np.log(x) - 1

def noise_floor(model_fn, F, y, n_repeats: int = 5, q: float = 95, seed: int = 42) -> dict:
    """The three lenses with a permuted-target control (rule T4). Fits on the first 80% in time order; permutation importance on the
       last 20% in R²_OS units; refits the same model on a PERMUTED target; the floor is the q-th percentile of |permutation importance|
       under permutation. Returns model, F_val, y_val, imp_real, perm_real, imp_null, perm_null, floor, r2_val.
       Beware: a 95th-percentile floor lets 1 feature in 20 through by chance — expect impostors among fifty features."""
    rng = np.random.default_rng(seed)
    cut = int(0.8 * len(F)); F_fit, y_fit, F_val, y_val = F.iloc[:cut], y.iloc[:cut], F.iloc[cut:], y.iloc[cut:]
    bench = ((y_val - y_fit.mean())**2).mean()
    def lenses(m):
        imp = pd.Series(getattr(m, "feature_importances_", np.full(F.shape[1], np.nan)), index=F.columns)
        pi  = permutation_importance(m, F_val, y_val, scoring="neg_mean_squared_error", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
        return imp, pd.Series(pi.importances_mean / bench, index=F.columns), pi.importances / bench
    m_real = model_fn().fit(F_fit, y_fit);  imp_real, perm_real, _        = lenses(m_real)
    y_perm = pd.Series(rng.permutation(y_fit.values), index=y_fit.index)
    m_null = model_fn().fit(F_fit, y_perm); imp_null, perm_null, raw_null = lenses(m_null)
    return dict(model=m_real, F_val=F_val, y_val=y_val, imp_real=imp_real, perm_real=perm_real, imp_null=imp_null, perm_null=perm_null,
                floor=float(np.percentile(np.abs(raw_null), q)), r2_val=r2_os(y_val, m_real.predict(F_val), y_fit.mean()))

def business_day_panel(prices: dict, calendar_stem: str = "spy", test_start=TEST_START) -> tuple[pd.DataFrame, pd.Series]:
    """Holdout log returns (%) of every asset on ONE calendar (the calendar_stem's business days), weekend returns accrued to the next
       business day through an as-of merge of the log price. Returns (R, days) where days = calendar days each row spans (3 on a Monday):
       scale a daily VARIANCE forecast of a 7-day asset by it. The forecast merge onto this calendar is yours to write — as-of, ffill."""
    cal_all = prices[calendar_stem].index; cal = cal_all[cal_all >= test_start]
    R = pd.DataFrame({s: 100 * np.log(prices[s]).reindex(cal_all, method="ffill").diff() for s in prices}).loc[cal]
    days = pd.Series(cal, index=cal).diff().dt.days.fillna(1)
    return R, days

def perf(port: pd.Series, W: pd.DataFrame, periods: int = 252) -> dict:
    """The Risk Agent's scoreboard for a daily log-return series (%) and its weight path: return, vol, Sharpe, max drawdown, turnover."""
    cum = port.cumsum()
    return {"ann. return %": port.mean() * periods, "ann. vol %": port.std() * np.sqrt(periods), "Sharpe": port.mean() / port.std() * np.sqrt(periods),
            "max drawdown %": (cum - cum.cummax()).min(), "turnover %/day": 100 * W.diff().abs().sum(axis=1).mean()}

def report(title: str, rules: list, disclosures: list = ()) -> dict:
    """Print an eval-suite report: rules is a list of (label, passed: bool). Returns {label[:2]: passed}. Your suite is the list."""
    print(title); print("=" * max(60, len(title)))
    ok = {}
    for label, passed in rules:
        ok[label[:2]] = bool(passed); print(f"  {'✅ PASS' if passed else '❌ FAIL'}  {label}")
    for line in disclosures:
        print(f"           ↳ {line}")
    print("=" * max(60, len(title)))
    return ok

TOY   = pd.DataFrame({"x_a": [0.02, 0.03, 0.03, 0.00], "x_b": [0.01, 0.04, 0.02, 0.03], "y": [0.015, 0.035, 0.035, 0.005]})
X_NEW = pd.DataFrame({"x_a": [0.024], "x_b": [0.030]})            # the new point; kept clear of the 0.025 cut

def agent_draft_tree_portfolio(ret: dict, prices: dict, stems=("spy", "eurusd", "btc"), close_utc: dict = DEFAULT_CLOSE_UTC) -> pd.Series:
    """DO NOT TRUST. A tree-based portfolio pipeline as an eager junior agent might write it: tunes a Random Forest per asset with
       GridSearchCV, names a 'top driver', turns predicted Sharpe ratios into softmax weights and backtests. It runs without error and
       is wrong in at least six distinct ways — leaks, fold errors, benchmark errors and the oldest alignment mistake in backtesting.
       Story S7: find them, quantify them, fix them."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    preds, vols, notes = {}, {}, []
    for stem in stems:
        r = ret[stem]
        X = feature_factory(stem, ret, prices, close_utc)
        X["trend5"] = r.rolling(5, center=True).mean()                                 # a smoother 5-day trend (centred = less lag)
        y = r.shift(-1)                                                                 # tomorrow's return
        keep = X.dropna().index.intersection(y.dropna().index); X, y = X.loc[keep], y.loc[keep]
        gs = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                          {"max_depth": [6, None], "max_features": ["sqrt", 0.5]}, cv=5, scoring="r2").fit(X, y)   # tune on all the data we have
        rf = gs.best_estimator_
        preds[stem] = pd.Series(rf.predict(X), index=X.index)                          # predicted next-day return
        vols[stem]  = r.rolling(21).std().reindex(X.index)
        notes.append(f"{stem}: train R² {rf.score(X, y):.3f}, CV R² {gs.best_score_:+.3f}, best {gs.best_params_}, top driver '{X.columns[np.argmax(rf.feature_importances_)]}'")
    P = pd.DataFrame(preds).dropna(); V = pd.DataFrame(vols).reindex(P.index)
    W = np.exp(P / V); W = W.div(W.sum(axis=1), axis=0)                                # softmax of predicted Sharpe → weights
    Rr = pd.DataFrame({s: ret[s] for s in stems}).reindex(P.index)
    port = (W * Rr).sum(axis=1)                                                         # backtest
    return pd.Series({"model": "RandomForest per asset, GridSearchCV", **{f"note {i+1}": n for i, n in enumerate(notes)},
                      "backtest days": len(port), "annualized Sharpe": port.mean() / port.std() * np.sqrt(252),
                      "total return %": port.sum(), "max weight on BTC": W["btc"].max() if "btc" in W else np.nan})