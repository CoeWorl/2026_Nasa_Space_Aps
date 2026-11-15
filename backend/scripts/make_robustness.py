"""Inferential robustness for the tide-backscatter result, for the atlas pages.

    uv run --project backend python -u backend/scripts/make_robustness.py \
        --app-dir data/processed/jamaica_bay/app

Run from the repository root. Reads only the exported app products
(timeseries.json, acquisitions.json, environment.json, models.json) and writes
robustness.json beside them. It computes no new measurement: every number here
is a different way of asking how much to trust the slopes that tide_qc.py and
tide_adjust.py already estimated.

What it adds, and why each one is here:

  ols / hc3        The published confidence intervals are classical OLS. These
                   observations are a time series with unequal spacing, so the
                   heteroskedasticity-consistent HC3 interval is reported beside
                   them. Where the two disagree, the wider one is the honest one.
  bootstrap        A pairs bootstrap of the slope, which assumes neither normal
                   errors nor constant variance.
  leave-one-out    One slope per dropped date, with its DFBETA. A result that
                   survives every single-date deletion is not one observation.
  influence        Leverage, studentised residual and Cook's distance per date,
                   so an outlier can be named instead of noticed.
  residuals        Durbin-Watson and lag-1 autocorrelation on the residuals in
                   date order, which is the assumption most at risk here.
  coverage sweep   The slope as a function of the minimum coverage threshold,
                   50% to 100%. The published 90% rule was chosen after the
                   ascending outlier was seen, so the whole curve is shown
                   rather than the one cut that was taken.
  interaction      A pooled diagnostic model, Y = a + b*h + c*D + d*(h*D), that
                   asks whether the descending and ascending slopes differ by
                   more than their uncertainty. Diagnostic only: the two tracks
                   have different baselines and geometry, so the direction-
                   specific models stay primary.
  per-region       Each marsh polygon's own slope with an HC3 interval,
                   Benjamini-Hochberg q-values across polygons, and a
                   random-effects (DerSimonian-Laird) partial-pooling estimate
                   with shrunken per-polygon slopes. Shrinkage is what makes a
                   polygon map defensible: 45 independent tests are not.
  permutation      The polygon slopes are not independent, because every polygon
                   sees the same acquisitions. Water level is permuted across
                   acquisition dates, keeping each date's whole spatial vector
                   intact, and all polygon slopes are refitted. This preserves
                   the cross-polygon dependence that a binomial test ignores.
  gauges           The spread between the two USGS gauges at each overpass, and
                   the reading offsets, since h is their mean and the tide-match
                   tolerances are 0.05 to 0.20 m.
  upland           Whether the upland reference itself follows water level. The
                   marsh-minus-upland response is only justified if it does not.
  response         Held-out error for marsh HH alone against marsh minus upland,
                   on the same chronological split, so the choice of response is
                   a measured one.

Nothing here is smoothed, imputed or pooled across tracks except the one model
explicitly labelled a pooled diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

BOOTSTRAP_DRAWS = 10_000
PERMUTATION_DRAWS = 2_000
SEED = 20261114
COVERAGE_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
TRACKS = ("descending_069", "ascending_133")


# --- least squares -----------------------------------------------------------


def _design(*columns: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(columns[0])), *columns])


def ols(x_design: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """OLS with classical and HC3 standard errors, plus hat values."""
    n, k = x_design.shape
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    beta = xtx_inv @ x_design.T @ y
    resid = y - x_design @ beta
    dof = n - k
    sigma2 = float(resid @ resid / dof)
    cov_ols = sigma2 * xtx_inv
    hat = np.einsum("ij,jk,ik->i", x_design, xtx_inv, x_design)
    omega = (resid / np.clip(1.0 - hat, 1e-12, None)) ** 2
    cov_hc3 = xtx_inv @ (x_design.T * omega) @ x_design @ xtx_inv
    se_ols = np.sqrt(np.diag(cov_ols))
    se_hc3 = np.sqrt(np.diag(cov_hc3))
    crit = float(stats.t.ppf(0.975, dof))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hc3 = beta / se_hc3
    p_hc3 = 2.0 * stats.t.sf(np.abs(t_hc3), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(n),
        "beta": beta,
        "resid": resid,
        "hat": hat,
        "dof": int(dof),
        "se_ols": se_ols,
        "se_hc3": se_hc3,
        "t_hc3": t_hc3,
        "p_hc3": p_hc3,
        "crit": crit,
        "sigma": math.sqrt(sigma2),
        "r2": float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def simple_slope(h: np.ndarray, y: np.ndarray) -> float:
    hc = h - h.mean()
    denom = float(hc @ hc)
    return float(hc @ (y - y.mean()) / denom) if denom > 0 else float("nan")


def coefficient(fit: dict[str, Any], index: int) -> dict[str, Any]:
    beta = float(fit["beta"][index])
    se_o = float(fit["se_ols"][index])
    se_h = float(fit["se_hc3"][index])
    crit = fit["crit"]
    return {
        "estimate": beta,
        "se_ols": se_o,
        "ci95_ols": [beta - crit * se_o, beta + crit * se_o],
        "se_hc3": se_h,
        "ci95_hc3": [beta - crit * se_h, beta + crit * se_h],
        "t_hc3": float(fit["t_hc3"][index]),
        "p_hc3": float(fit["p_hc3"][index]),
    }


# --- single-track robustness --------------------------------------------------


def bootstrap_slope(h: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = len(h)
    idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    hs, ys = h[idx], y[idx]
    hc = hs - hs.mean(axis=1, keepdims=True)
    denom = (hc**2).sum(axis=1)
    good = denom > 0
    slopes = np.full(BOOTSTRAP_DRAWS, np.nan)
    slopes[good] = (hc[good] * (ys[good] - ys[good].mean(axis=1, keepdims=True))).sum(
        axis=1
    ) / denom[good]
    slopes = slopes[np.isfinite(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {
        "draws": int(slopes.size),
        "method": "pairs bootstrap, percentile interval",
        "slope_median": float(np.median(slopes)),
        "slope_ci95": [float(lo), float(hi)],
        "se": float(slopes.std(ddof=1)),
        "fraction_negative": float((slopes < 0).mean()),
    }


def leave_one_out(dates: list[str], h: np.ndarray, y: np.ndarray, full: float) -> list[dict]:
    out = []
    for i, date in enumerate(dates):
        keep = np.ones(len(h), dtype=bool)
        keep[i] = False
        slope = simple_slope(h[keep], y[keep])
        out.append(
            {
                "date": date,
                "slope_db_per_m": slope,
                "dfbeta_db_per_m": slope - full,
            }
        )
    return out


def influence(dates: list[str], h: np.ndarray, y: np.ndarray, fit: dict[str, Any]) -> list[dict]:
    resid, hat, sigma = fit["resid"], fit["hat"], fit["sigma"]
    fitted = y - resid
    denom = sigma * np.sqrt(np.clip(1.0 - hat, 1e-12, None))
    std_resid = resid / denom
    cooks = (std_resid**2 / 2.0) * (hat / np.clip(1.0 - hat, 1e-12, None))
    return [
        {
            "date": date,
            "h_m": float(h[i]),
            "y_db": float(y[i]),
            "fitted_db": float(fitted[i]),
            "residual_db": float(resid[i]),
            "standardised_residual": float(std_resid[i]),
            "leverage": float(hat[i]),
            "cooks_distance": float(cooks[i]),
        }
        for i, date in enumerate(dates)
    ]


def residual_diagnostics(resid: np.ndarray) -> dict[str, Any]:
    diff = np.diff(resid)
    dw = float((diff @ diff) / (resid @ resid))
    centred = resid - resid.mean()
    denom = float(centred @ centred)
    lag1 = float(centred[1:] @ centred[:-1] / denom) if denom > 0 else float("nan")
    return {
        "note": "residuals in date order, from the slope fit on this variant",
        "durbin_watson": dw,
        "lag1_autocorrelation": lag1,
        "shapiro_p": float(stats.shapiro(resid).pvalue) if len(resid) >= 3 else None,
        "n": int(len(resid)),
    }


def coverage_sweep(rows: list[dict]) -> list[dict]:
    """Slope as a function of the minimum coverage kept, not just the 90% cut."""
    out = []
    for threshold in COVERAGE_GRID:
        kept = [r for r in rows if min(r["marsh_coverage"], r["upland_coverage"]) >= threshold]
        entry: dict[str, Any] = {
            "min_coverage": threshold,
            "n": len(kept),
            "excluded_dates": [
                r["date"]
                for r in rows
                if min(r["marsh_coverage"], r["upland_coverage"]) < threshold
            ],
        }
        if len(kept) >= 4:
            h = np.array([r["h"] for r in kept])
            y = np.array([r["y"] for r in kept])
            fit = ols(_design(h), y)
            entry.update(
                {
                    "slope_db_per_m": float(fit["beta"][1]),
                    "ci95_ols": coefficient(fit, 1)["ci95_ols"],
                    "ci95_hc3": coefficient(fit, 1)["ci95_hc3"],
                }
            )
        else:
            entry.update({"slope_db_per_m": None, "ci95_ols": None, "ci95_hc3": None})
        out.append(entry)
    return out


def track_variant(dates: list[str], h: np.ndarray, y: np.ndarray, rng) -> dict[str, Any]:
    fit = ols(_design(h), y)
    slope = float(fit["beta"][1])
    loo = leave_one_out(dates, h, y, slope)
    loo_slopes = [r["slope_db_per_m"] for r in loo]
    worst = max(loo, key=lambda r: abs(r["dfbeta_db_per_m"]))
    theil = stats.theilslopes(y, h, 0.95)
    return {
        "n": int(len(h)),
        "date_range": [dates[0], dates[-1]],
        "ols": {
            "intercept_db": float(fit["beta"][0]),
            "slope": coefficient(fit, 1),
            "r2": fit["r2"],
            "residual_sd_db": fit["sigma"],
        },
        "theil_sen": {
            "slope_db_per_m": float(theil[0]),
            "ci95": [float(theil[2]), float(theil[3])],
        },
        "bootstrap": bootstrap_slope(h, y, rng),
        "leave_one_out": {
            "slopes": loo,
            "range": [float(min(loo_slopes)), float(max(loo_slopes))],
            "all_negative": bool(all(s < 0 for s in loo_slopes)),
            "most_influential_date": worst["date"],
            "largest_dfbeta_db_per_m": worst["dfbeta_db_per_m"],
        },
        "influence": influence(dates, h, y, fit),
        "residual_diagnostics": residual_diagnostics(fit["resid"]),
    }


# --- per-polygon --------------------------------------------------------------


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def dersimonian_laird(beta: np.ndarray, var: np.ndarray) -> dict[str, Any]:
    """Random-effects pooling: a mean slope, a between-polygon spread, and the
    shrunken polygon estimates that follow from them."""
    w = 1.0 / var
    mu_fixed = float((w * beta).sum() / w.sum())
    q = float((w * (beta - mu_fixed) ** 2).sum())
    k = len(beta)
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_re = 1.0 / (var + tau2)
    mu = float((w_re * beta).sum() / w_re.sum())
    se_mu = float(math.sqrt(1.0 / w_re.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {
        "k": k,
        "mu_db_per_m": mu,
        "mu_se": se_mu,
        "mu_ci95": [mu - 1.96 * se_mu, mu + 1.96 * se_mu],
        "tau_db_per_m": math.sqrt(tau2),
        "q": q,
        "q_p": float(stats.chi2.sf(q, k - 1)),
        "i_squared": i2,
        "method": "DerSimonian-Laird random effects over polygon slopes",
    }


def shrink(beta: float, var: float, mu: float, tau2: float) -> dict[str, Any]:
    if tau2 <= 0:
        post_mean, post_var = mu, var * 0.0 + 1e-12
    else:
        prec = 1.0 / var + 1.0 / tau2
        post_mean = (beta / var + mu / tau2) / prec
        post_var = 1.0 / prec
    se = math.sqrt(post_var)
    return {
        "slope_db_per_m": post_mean,
        "se": se,
        "ci95": [post_mean - 1.96 * se, post_mean + 1.96 * se],
        "prob_negative": float(stats.norm.cdf(0.0, loc=post_mean, scale=max(se, 1e-9))),
    }


def region_permutation(h: np.ndarray, ys: list[np.ndarray], masks: list[np.ndarray], rng):
    """Permute water level across acquisition dates, keeping each date's whole
    spatial vector together, and refit every polygon slope."""
    n_dates = len(h)
    observed = np.array([simple_slope(h[m], y) for y, m in zip(ys, masks)])
    n_neg = np.zeros(PERMUTATION_DRAWS, dtype=int)
    med = np.zeros(PERMUTATION_DRAWS)
    perms = np.array([rng.permutation(n_dates) for _ in range(PERMUTATION_DRAWS)])
    for b in range(PERMUTATION_DRAWS):
        hp = h[perms[b]]
        slopes = np.array([simple_slope(hp[m], y) for y, m in zip(ys, masks)])
        n_neg[b] = int((slopes < 0).sum())
        med[b] = float(np.median(slopes))
    obs_neg = int((observed < 0).sum())
    obs_med = float(np.median(observed))
    return {
        "draws": PERMUTATION_DRAWS,
        "unit": "acquisition date; the whole polygon vector moves together",
        "observed_n_negative": obs_neg,
        "n_polygons": len(observed),
        "p_n_negative": float(((n_neg >= obs_neg).sum() + 1) / (PERMUTATION_DRAWS + 1)),
        "null_n_negative_mean": float(n_neg.mean()),
        "null_n_negative_95": float(np.percentile(n_neg, 95)),
        "observed_median_slope": obs_med,
        "p_median_slope": float(((med <= obs_med).sum() + 1) / (PERMUTATION_DRAWS + 1)),
        "null_median_slope_ci95": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
    }


def per_region(track_regions: dict, acq_ids: list[str], h_by_id: dict, keep_ids: set, rng):
    index = [i for i, a in enumerate(acq_ids) if a in keep_ids]
    fits, ys, masks = [], [], []
    h_all = np.array([h_by_id[acq_ids[i]] for i in index])
    for region in track_regions:
        values = region["marsh_relative_db"]
        picked = [(k, values[i]) for k, i in enumerate(index) if values[i] is not None]
        if len(picked) < 6:
            continue
        mask = np.zeros(len(index), dtype=bool)
        mask[[k for k, _ in picked]] = True
        y = np.array([v for _, v in picked], dtype=float)
        h = h_all[mask]
        fit = ols(_design(h), y)
        coef = coefficient(fit, 1)
        fits.append(
            {
                "region_id": region["region_id"],
                "kind": region.get("kind"),
                "n": int(len(y)),
                "n_pixels": region.get("n_pixels"),
                "slope_db_per_m": coef["estimate"],
                "se_hc3": coef["se_hc3"],
                "ci95_hc3": coef["ci95_hc3"],
                "p_uncorrected": coef["p_hc3"],
            }
        )
        ys.append(y)
        masks.append(mask)
    if not fits:
        return None
    p = np.array([f["p_uncorrected"] for f in fits])
    q = benjamini_hochberg(p)
    beta = np.array([f["slope_db_per_m"] for f in fits])
    var = np.array([max(f["se_hc3"], 1e-6) ** 2 for f in fits])
    pooled = dersimonian_laird(beta, var)
    tau2 = pooled["tau_db_per_m"] ** 2
    for i, f in enumerate(fits):
        f["q_bh"] = float(q[i])
        f["partial_pooled"] = shrink(beta[i], var[i], pooled["mu_db_per_m"], tau2)
    return {
        "n_polygons": len(fits),
        "n_negative": int((beta < 0).sum()),
        "n_p05_uncorrected": int((p < 0.05).sum()),
        "n_q05_bh": int((q < 0.05).sum()),
        "n_shrunk_prob_negative_95": sum(
            1 for f in fits if f["partial_pooled"]["prob_negative"] >= 0.95
        ),
        "expected_false_positives_at_5pc": round(0.05 * len(fits), 2),
        "pooled": pooled,
        "permutation": region_permutation(h_all, ys, masks, rng),
        "fits": sorted(fits, key=lambda f: f["slope_db_per_m"]),
    }


# --- small extras -------------------------------------------------------------


def gauge_agreement(observations: list[dict]) -> dict[str, Any]:
    rows = []
    for o in observations:
        a = o.get("usgs_inwood_water_level_m_navd88")
        b = o.get("usgs_rockaway_water_level_m_navd88")
        if a is None or b is None:
            continue
        offsets = [abs(v) for v in (o.get("time_offset_seconds") or {}).values()]
        rows.append(
            {
                "acquisition_id": o["acquisition_id"],
                "difference_m": float(a - b),
                "mean_m": float(o["mean_water_level_m_navd88"]),
                "max_offset_s": max(offsets) if offsets else None,
            }
        )
    diffs = np.array([abs(r["difference_m"]) for r in rows])
    offs = np.array([r["max_offset_s"] for r in rows if r["max_offset_s"] is not None])
    return {
        "n": len(rows),
        "definition": "Inwood minus Rockaway at the overpass; h is their mean",
        "median_abs_difference_m": float(np.median(diffs)),
        "p90_abs_difference_m": float(np.percentile(diffs, 90)),
        "max_abs_difference_m": float(diffs.max()),
        "n_above_0_05_m": int((diffs > 0.05).sum()),
        "n_above_0_10_m": int((diffs > 0.10).sum()),
        "median_offset_s": float(np.median(offs)) if offs.size else None,
        "max_offset_s": float(offs.max()) if offs.size else None,
        "per_date": rows,
    }


def upland_stability(dates, h, upland, marsh) -> dict[str, Any]:
    fit_u = ols(_design(h), upland)
    fit_m = ols(_design(h), marsh)
    return {
        "note": "marsh minus upland is only justified if the upland reference "
        "does not itself follow water level",
        "upland_slope": coefficient(fit_u, 1),
        "upland_sd_db": float(np.std(upland, ddof=1)),
        "marsh_only_slope": coefficient(fit_m, 1),
        "series": [
            {"date": d, "upland_hh_db": float(u), "marsh_hh_db": float(m), "h_m": float(x)}
            for d, u, m, x in zip(dates, upland, marsh, h)
        ],
    }


def heldout_rmse(h: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Chronological half split: fit on the first half, score on the second."""
    cut = len(h) // 2
    h_tr, y_tr, h_te, y_te = h[:cut], y[:cut], h[cut:], y[cut:]
    beta = simple_slope(h_tr, y_tr)
    alpha = float(y_tr.mean() - beta * h_tr.mean())
    pred = alpha + beta * h_te
    const = float(y_tr.mean())
    rmse = lambda e: float(np.sqrt(np.mean(e**2)))  # noqa: E731
    return {
        "n_test": int(len(y_te)),
        "constant_rmse_db": rmse(y_te - const),
        "tide_rmse_db": rmse(y_te - pred),
        "tide_mae_db": float(np.mean(np.abs(y_te - pred))),
        "improvement_vs_constant": 1.0 - rmse(y_te - pred) / rmse(y_te - const),
        "beta_train_db_per_m": beta,
    }


# --- main ---------------------------------------------------------------------


def build(app_dir: Path) -> dict[str, Any]:
    series = json.loads((app_dir / "timeseries.json").read_text())
    acqs = json.loads((app_dir / "acquisitions.json").read_text())
    env = json.loads((app_dir / "environment.json").read_text())
    rng = np.random.default_rng(SEED)

    acq_by_id = {a["acquisition_id"]: a for a in acqs["acquisitions"]}
    out: dict[str, Any] = {
        "meta": {
            "generated_from": "app products only; no new measurement",
            "spread_definition": "standard deviation in dB, ddof=1",
            "error_metric": "RMSE in dB unless named otherwise",
            "hc3": "heteroskedasticity-consistent (MacKinnon-White) standard errors",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "permutation_draws": PERMUTATION_DRAWS,
            "seed": SEED,
            "coverage_grid": COVERAGE_GRID,
            "primary_response": "Y = median marsh HH (dB) minus median upland HH (dB)",
        },
        "tracks": {},
        "regions": {},
    }

    pooled_rows = []
    for track in TRACKS:
        rows = []
        for p in series["pooled"][track]:
            a = acq_by_id.get(p["acquisition_id"], {})
            rows.append(
                {
                    "date": p["date"],
                    "h": p["mean_water_level_m_navd88"],
                    "y": p["marsh_relative_db"],
                    "marsh_hh": p["marsh_hh_db"],
                    "upland_hh": p["upland_hh_db"],
                    "ok": p["passes_coverage_screen"],
                    "marsh_coverage": a.get("marsh_coverage_fraction", 1.0),
                    "upland_coverage": a.get("upland_coverage_fraction", 1.0),
                    "acquisition_id": p["acquisition_id"],
                }
            )
        rows.sort(key=lambda r: r["date"])

        variants = {"all_dates": rows}
        if any(not r["ok"] for r in rows):
            variants["coverage_screened"] = [r for r in rows if r["ok"]]

        track_out: dict[str, Any] = {"variants": {}}
        for name, sub in variants.items():
            dates = [r["date"] for r in sub]
            h = np.array([r["h"] for r in sub])
            y = np.array([r["y"] for r in sub])
            track_out["variants"][name] = track_variant(dates, h, y, rng)

        primary = variants.get("coverage_screened", rows)
        h_p = np.array([r["h"] for r in primary])
        track_out["coverage_sweep"] = coverage_sweep(rows)
        track_out["upland"] = upland_stability(
            [r["date"] for r in primary],
            h_p,
            np.array([r["upland_hh"] for r in primary]),
            np.array([r["marsh_hh"] for r in primary]),
        )
        track_out["response_choice"] = {
            "marsh_minus_upland": heldout_rmse(h_p, np.array([r["y"] for r in primary])),
            "marsh_only": heldout_rmse(h_p, np.array([r["marsh_hh"] for r in primary])),
            "note": "same chronological split and the same dates; only the response differs",
        }
        out["tracks"][track] = track_out
        pooled_rows.extend((track, r) for r in primary)

    # Pooled interaction diagnostic.
    h = np.array([r["h"] for _, r in pooled_rows])
    y = np.array([r["y"] for _, r in pooled_rows])
    d = np.array([1.0 if t == "ascending_133" else 0.0 for t, _ in pooled_rows])
    fit = ols(_design(h, d, h * d), y)
    out["pooled_interaction"] = {
        "model": "Y = b0 + b1*h + b2*D + b3*(h*D), D = 1 for ascending",
        "status": "diagnostic only; the direction-specific models stay primary",
        "n": int(len(y)),
        "slope_descending": coefficient(fit, 1),
        "direction_offset_db": coefficient(fit, 2),
        "slope_difference": coefficient(fit, 3),
        "note": "b3 is the ascending slope minus the descending slope. The "
        "two tracks have different baselines and geometry, so a difference "
        "here is not by itself a physical difference in response.",
    }

    # Per-polygon.
    for track in TRACKS:
        block = series["regions"][track]
        acq_ids = block["acquisition_ids"]
        h_by_id = {
            p["acquisition_id"]: p["mean_water_level_m_navd88"] for p in series["pooled"][track]
        }
        keep = {
            p["acquisition_id"] for p in series["pooled"][track] if p["passes_coverage_screen"]
        }
        result = per_region(block["regions"], acq_ids, h_by_id, keep, rng)
        if result:
            result["screen"] = "coverage-screened acquisitions only"
            out["regions"][track] = result

    out["gauges"] = gauge_agreement(env["observations"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app-dir", type=Path, default=Path("data/processed/jamaica_bay/app"))
    args = ap.parse_args()
    app_dir = args.app_dir.resolve()
    if not (app_dir / "timeseries.json").is_file():
        raise SystemExit(f"no timeseries.json under {app_dir}; run tide_adjust.py first")
    payload = build(app_dir)
    target = app_dir / "robustness.json"
    target.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KiB)")
    for track, block in payload["tracks"].items():
        for name, v in block["variants"].items():
            s = v["ols"]["slope"]
            print(
                f"  {track}/{name}: n={v['n']} slope={s['estimate']:+.3f} "
                f"HC3 95% [{s['ci95_hc3'][0]:+.3f}, {s['ci95_hc3'][1]:+.3f}] "
                f"boot [{v['bootstrap']['slope_ci95'][0]:+.3f}, {v['bootstrap']['slope_ci95'][1]:+.3f}]"
            )
    for track, block in payload["regions"].items():
        p = block["permutation"]
        print(
            f"  {track}: {block['n_negative']}/{block['n_polygons']} negative "
            f"(null mean {p['null_n_negative_mean']:.1f}, p={p['p_n_negative']:.4f}), "
            f"BH q<0.05: {block['n_q05_bh']}, mu={block['pooled']['mu_db_per_m']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
