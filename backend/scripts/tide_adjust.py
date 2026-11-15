"""Does tide adjustment help on dates the model has not seen? Both tracks.

    uv run --project backend python -u backend/scripts/tide_adjust.py --out-dir data/processed/jamaica_bay/app

Run from the repository root. Reuses the frozen series definitions of tide_qc.py
(Y_t = median marsh HH dB minus median upland HH dB, h_t = mean of the two USGS gauges, m NAVD88).
Nothing here changes the frozen regressions; it evaluates them out of sample.

Per track and per series (all dates, and the post hoc >= 90% coverage screen):
  1. Three-model benchmark on held-out dates
       A constant   : training mean
       B seasonal   : intercept + sin/cos of day of year
       C tide       : intercept + beta * h
     under (i) the chronological half split and (ii) a rolling origin (fit on dates 1..k,
     predict date k+1, k >= MIN_TRAIN). RMSE, MAE, and 1 - RMSE_model / RMSE_constant.
  2. Tide-adjusted series  Y_adj = Y - beta * (h - H_REF)
       out of sample : beta from the first half, applied to the second half
       in sample     : beta from every date (descriptive only; labelled as such)
     with spread (SD, MAD) before and after and the residual correlation with h.
  3. Pairwise comparability on held-out dates: |dY| against |dh| for every pair, raw and
     adjusted. Pairs share dates, so this is descriptive, not n*(n-1)/2 independent tests.
  4. Injected step: a persistent +1/+2/+3 dB shift added to the last INJECT_LEN held-out dates.
     With beta frozen from training the adjustment cannot touch the step (it does not depend
     on Y), so recovery is exact by construction. The informative variant refits beta on all
     dates including the injected ones and reports how much of the step survives.

  5. Exploratory only: a linear trend term (Y = alpha + beta*h + gamma*t) and runs of same-sign
     residuals from the tide fit ("candidate periods"). Neither is evidence of ecological change.

Also exports the five data products for the frontend:
  acquisitions.json   one record per physical acquisition (highest processing version kept)
  environment.json    gauge levels, gauge time offsets, daily weather, quality flags
  timeseries.json     pooled marsh/upland series and one series per marsh polygon, HH and HV
  models.json         regressions, validation, trend and persistence, per orbit
  restoration_projects.geojson, regions.geojson, hydrographs.json
HV and number of looks come from stack_hv.npz (extract_hv.py); gauge curves and weather from
fetch_context.py. Both are optional: fields are null when the inputs are missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from tide_qc import COVERAGE_MIN, load_track, regress, to_db

H_REF = 0.0  # m NAVD88; inside the observed range of both tracks
MIN_TRAIN = 10
INJECT_LEN = 4
INJECT_DB = (1.0, 2.0, 3.0)
MIN_REGION_PIXELS = 20  # same floor as the exploratory per-polygon test in tide_regression.py
CRID_RE = re.compile(r"_(P\d{5})_")

TRACKS = {
    "descending_069": {"stack": "stack_bay_069D068", "tides": "jamaica_bay/tide_matches_27.csv",
                       "direction": "descending", "track": 69, "frame": 68},
    "ascending_133": {"stack": "stack_bay_133A022", "tides": "jamaica_bay/tide_matches_133A022.csv",
                      "direction": "ascending", "track": 133, "frame": 22},
}


def doy_terms(dates: list[str]) -> np.ndarray:
    d = np.array([date.fromisoformat(s).timetuple().tm_yday for s in dates])
    w = 2 * np.pi * d / 365.25
    return np.column_stack([np.sin(w), np.cos(w)])


def design(kind: str, h: np.ndarray, season: np.ndarray) -> np.ndarray:
    one = np.ones((len(h), 1))
    if kind == "constant":
        return one
    if kind == "seasonal":
        return np.hstack([one, season])
    return np.hstack([one, h[:, None]])


def predict(kind, h, season, y, tr, te) -> np.ndarray:
    X = design(kind, h, season)
    coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    return X[te] @ coef


def errors(obs: np.ndarray, pred: np.ndarray) -> dict:
    e = obs - pred
    return {"rmse_db": float(np.sqrt(np.mean(e**2))), "mae_db": float(np.mean(np.abs(e)))}


def benchmark(h, season, y, splits) -> dict:
    """splits: list of (train_idx, test_idx). Errors are pooled over every test date."""
    out = {}
    for kind in ("constant", "seasonal", "tide"):
        obs, pred = [], []
        for tr, te in splits:
            pred.append(predict(kind, h, season, y, tr, te))
            obs.append(y[te])
        out[kind] = errors(np.concatenate(obs), np.concatenate(pred))
    base = out["constant"]["rmse_db"]
    for kind in ("seasonal", "tide"):
        out[kind]["rmse_improvement_vs_constant"] = float(1 - out[kind]["rmse_db"] / base)
    out["n_test_predictions"] = int(sum(len(te) for _, te in splits))
    return out


def mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def spread(y, h) -> dict:
    r, p = stats.pearsonr(h, y)
    return {"sd_db": float(np.std(y, ddof=1)), "mad_db": mad(y), "r_with_water_level": float(r), "p": float(p)}


def pairwise(y, h) -> dict:
    i, j = np.triu_indices(len(y), k=1)
    dh, dy = np.abs(h[i] - h[j]), np.abs(y[i] - y[j])
    near = dh <= np.median(dh)
    return {"n_pairs": len(dh), "spearman_dh_dy": float(stats.spearmanr(dh, dy)[0]),
            "median_abs_dy_similar_tide_db": float(np.median(dy[near])),
            "median_abs_dy_different_tide_db": float(np.median(dy[~near])),
            "split_at_abs_dh_m": float(np.median(dh))}


def analyse(dates: list[str], h: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    half = n // 2
    tr, te = np.arange(half), np.arange(half, n)
    season = doy_terms(dates)

    rolling = [(np.arange(k), np.array([k])) for k in range(MIN_TRAIN, n)]
    bench = {"chronological_half": benchmark(h, season, y, [(tr, te)]),
             "rolling_origin": benchmark(h, season, y, rolling)}
    bench["chronological_half"]["train_dates"] = [dates[0], dates[half - 1]]
    bench["chronological_half"]["test_dates"] = [dates[half], dates[-1]]
    bench["rolling_origin"]["min_train"] = MIN_TRAIN

    b_train = float(np.polyfit(h[tr], y[tr], 1)[0])
    b_all = float(np.polyfit(h, y, 1)[0])
    adj_oos = y - b_train * (h - H_REF)  # only indices in `te` are out of sample
    adj_ins = y - b_all * (h - H_REF)

    adjustment = {
        "h_ref_m_navd88": H_REF,
        "beta_train_db_per_m": b_train,
        "beta_all_dates_db_per_m": b_all,
        "held_out": {"n": len(te), "raw": spread(y[te], h[te]), "adjusted": spread(adj_oos[te], h[te])},
        "in_sample_descriptive": {"n": n, "raw": spread(y, h), "adjusted": spread(adj_ins, h)},
        "pairwise_held_out": {"raw": pairwise(y[te], h[te]), "adjusted": pairwise(adj_oos[te], h[te]),
                              "note": "pairs share acquisitions; descriptive only"},
    }
    ho = adjustment["held_out"]
    ho["sd_reduction"] = float(1 - ho["adjusted"]["sd_db"] / ho["raw"]["sd_db"])

    # Injected persistent step on the last INJECT_LEN held-out dates.
    win = np.arange(n - INJECT_LEN, n)
    rest = np.setdiff1d(te, win)
    inject = []
    for amp in INJECT_DB:
        y2 = y.copy()
        y2[win] += amp

        def step(series):  # median of the window minus median of the other held-out dates
            return float(np.median(series[win]) - np.median(series[rest]))

        frozen = step(y2 - b_train * (h - H_REF)) - step(adj_oos)
        b_refit = float(np.polyfit(h, y2, 1)[0])
        refit = step(y2 - b_refit * (h - H_REF)) - step(y - b_all * (h - H_REF))
        inject.append({"injected_db": amp, "recovered_frozen_beta_db": frozen,
                       "recovered_refit_beta_db": refit, "refit_beta_db_per_m": b_refit})
    adjustment["injected_step"] = {
        "window_dates": [dates[i] for i in win], "results": inject,
        "note": "frozen-beta recovery is exact by construction; the refit column is the informative one",
    }

    per_date = [{"date": d, "y_db": float(a), "h_m": float(b),
                 "y_adj_out_of_sample_db": float(c) if k >= half else None,
                 "y_adj_in_sample_db": float(e), "role": "test" if k >= half else "train"}
                for k, (d, a, b, c, e) in enumerate(zip(dates, y, h, adj_oos, adj_ins))]
    reg = regress(y, h)
    reg["intercept_db"] = float(np.polyfit(h, y, 1)[1])
    return {"n": n, "regression": reg, "benchmark": bench, "tide_adjustment": adjustment,
            "trend": trend(dates, h, y), "persistence": persistence(dates, h, y), "per_date": per_date}


def verdict(a: dict) -> str:
    """Strong / mixed / weak, by rules written before looking at the numbers."""
    half = a["benchmark"]["chronological_half"]["tide"]["rmse_improvement_vs_constant"]
    roll = a["benchmark"]["rolling_origin"]["tide"]["rmse_improvement_vs_constant"]
    ho = a["tide_adjustment"]["held_out"]
    resid_ok = abs(ho["adjusted"]["r_with_water_level"]) < abs(ho["raw"]["r_with_water_level"])
    keeps = all(abs(r["recovered_refit_beta_db"] - r["injected_db"]) <= 0.25 * r["injected_db"]
                for r in a["tide_adjustment"]["injected_step"]["results"])
    wins = sum([half > 0.1, roll > 0.1, resid_ok and ho["sd_reduction"] > 0, keeps])
    return "strong" if wins == 4 else "mixed" if wins >= 2 else "weak"


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, standard errors, t-based 95% half-widths, two-sided p-values."""
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se = np.sqrt(np.diag((resid @ resid) / (n - k) * np.linalg.inv(X.T @ X)))
    half = stats.t.ppf(0.975, n - k) * se
    p = 2 * stats.t.sf(np.abs(coef / se), n - k)
    return coef, se, half, p


def trend(dates: list[str], h: np.ndarray, y: np.ndarray) -> dict:
    """Y = alpha + beta*h + gamma*t. Exploratory: about one year of data, so gamma and season are confounded."""
    t = np.array([(date.fromisoformat(d) - date.fromisoformat(dates[0])).days / 365.25 for d in dates])
    coef, _, half, p = ols(np.column_stack([np.ones_like(h), h, t]), y)
    return {
        "model": "Y = alpha + beta*h + gamma*t, t in years since the first date",
        "gamma_db_per_year": float(coef[2]),
        "gamma_95ci_db_per_year": [float(coef[2] - half[2]), float(coef[2] + half[2])],
        "gamma_p": float(p[2]),
        "beta_with_trend_db_per_m": float(coef[1]),
        "record_length_years": float(t[-1]),
        "status": "exploratory",
        "note": "A trend in radar backscatter in dB per year. Not a subsidence, elevation-change or land-loss rate, "
                "and not separable from seasonality with about one year of data.",
    }


def persistence(dates: list[str], h: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> dict:
    """Do residuals from the tide fit stay on one side of zero for several dates in a row?"""
    slope, icpt = np.polyfit(h, y, 1)
    resid = y - (icpt + slope * h)
    scale = 1.4826 * mad(resid)
    z = resid / scale
    sign = np.sign(resid)

    def runs(s: np.ndarray) -> list[tuple[int, int]]:
        out, start = [], 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[start]:
                out.append((start, i - 1))
                start = i
        return out

    def longest(s: np.ndarray) -> int:
        return max(b - a + 1 for a, b in runs(s))

    observed = longest(sign)
    rng = np.random.default_rng(20261114)
    p_run = float(np.mean([longest(rng.permutation(sign)) >= observed for _ in range(n_perm)]))
    candidates = [{"start": dates[a], "end": dates[b], "n_dates": b - a + 1,
                   "direction": "above fit" if sign[a] > 0 else "below fit",
                   "median_residual_db": float(np.median(resid[a:b + 1]))}
                  for a, b in runs(sign) if b - a + 1 >= 3]
    return {
        "residual_definition": "Y minus the all-date tide fit (in sample)",
        "robust_scale_db": float(scale),
        "lag1_autocorrelation": float(stats.pearsonr(resid[:-1], resid[1:])[0]),
        "longest_same_sign_run": int(observed),
        "longest_run_permutation_p": p_run,
        "candidate_periods": candidates,
        "single_date_outliers": [{"date": d, "residual_db": float(r), "robust_z": float(v)}
                                 for d, r, v in zip(dates, resid, z) if abs(v) > 2],
        "per_date": [{"date": d, "residual_db": float(r), "robust_z": float(v)} for d, r, v in zip(dates, resid, z)],
        "status": "exploratory",
        "note": "Candidate periods are runs of 3 or more consecutive dates on one side of the fit. They mark where to "
                "look, not confirmed ecological change; acquisitions are 12 days apart and seasonal effects are not removed.",
    }


def num(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def region_stats(stack_db: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median of per-pixel dB over a region for every date (NaN below MIN_REGION_PIXELS) and the valid count."""
    vals = stack_db.reshape(stack_db.shape[0], -1)[:, idx]
    count = np.isfinite(vals).sum(axis=1)
    med = np.full(len(vals), np.nan)
    for t in np.where(count >= MIN_REGION_PIXELS)[0]:
        med[t] = np.nanmedian(vals[t])
    return med, count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--polygons", type=Path, default=Path("data/processed/jamaica_bay/marsh_polygons_osm.geojson"))
    ap.add_argument("--projects", type=Path, default=Path("configs/restoration_projects.geojson"))
    args = ap.parse_args()
    base = Path("data/processed")
    bay = base / "jamaica_bay"
    weather = json.loads((bay / "weather.json").read_text())["acquisitions"] if (bay / "weather.json").exists() else {}

    acquisitions, environment, pooled, regions, models = [], [], {}, {}, {}
    region_features: dict[str, dict] = {}
    for name, cfg in TRACKS.items():
        tr = load_track(base / cfg["stack"], base / cfg["tides"], args.polygons)
        tide_rows = {r["product_id"]: r for r in csv.DictReader((base / cfg["tides"]).open())}
        dates, h, db = tr["dates"], tr["h"], tr["db"]
        n_t = len(dates)
        acq_ids = [f"{name}_{d}" for d in dates]
        flat = lambda mask: np.flatnonzero(mask.ravel())
        marsh_idx, upland_idx = flat(tr["marsh"]), flat(tr["upland"])

        hv_path = base / cfg["stack"] / "stack_hv.npz"
        hv_db = looks = None
        if hv_path.exists():
            zh = np.load(hv_path)
            if [str(p) for p in zh["product_ids"][tr["sel"]]] != tr["product_ids"]:
                raise SystemExit(f"{hv_path} does not match the HH stack")
            hv_db = np.stack([to_db(zh["HV_values"][i], zh["HV_valid"][i]) for i in tr["sel"]])
            looks = zh["looks"][tr["sel"]].astype(np.float32)

        mdb = np.array([np.nanmedian(db[t][tr["marsh"]]) for t in range(n_t)])
        udb = np.array([np.nanmedian(db[t][tr["upland"]]) for t in range(n_t)])
        y = mdb - udb
        ok = (tr["cov_m"] >= COVERAGE_MIN) & (tr["cov_u"] >= COVERAGE_MIN)
        nan = np.full(n_t, np.nan)
        m_hv, _ = region_stats(hv_db, marsh_idx) if hv_db is not None else (nan, nan)
        u_hv, _ = region_stats(hv_db, upland_idx) if hv_db is not None else (nan, nan)
        y_hv = m_hv - u_hv

        # Products 1 and 2: one record per physical acquisition.
        for t, (d, pid) in enumerate(zip(dates, tr["product_ids"])):
            r = tide_rows[pid]
            crid = CRID_RE.search(pid)
            acquisitions.append({
                "acquisition_id": acq_ids[t], "product_id": pid, "timestamp_utc": r["acquired_utc"],
                "orbit_direction": cfg["direction"], "track": cfg["track"], "frame": cfg["frame"],
                "polarizations": ["HH", "HV"] if hv_db is not None else ["HH"],
                "processing_version": pid[-3:], "crid": crid.group(1) if crid else None,
                "quality_valid_fraction": float(tr["valid"][t].mean()),
                "marsh_coverage_fraction": float(tr["cov_m"][t]),
                "upland_coverage_fraction": float(tr["cov_u"][t]),
                "passes_coverage_screen": bool(ok[t]),
            })
            offs = {g: float(r[f"{g}_offset_min"]) * 60.0 for g in ("inwood", "rockaway")}
            lv = {g: float(r[f"{g}_m"]) for g in ("inwood", "rockaway")}
            flags = [f"{g}_gauge_{r[f'{g}_status'].lower()}" for g in lv if r[f"{g}_status"] != "Approved"]
            flags += [f"{g}_gauge_offset_over_6min" for g in lv if abs(offs[g]) > 360]
            if abs(lv["inwood"] - lv["rockaway"]) > 0.25:
                flags.append("gauges_differ_over_0.25m")
            if not ok[t]:
                flags.append("partial_radar_coverage")
            w = weather.get(acq_ids[t], {})
            environment.append({
                "acquisition_id": acq_ids[t],
                "usgs_inwood_water_level_m_navd88": lv["inwood"], "usgs_rockaway_water_level_m_navd88": lv["rockaway"],
                "mean_water_level_m_navd88": float(h[t]), "time_offset_seconds": offs,
                "water_level_kind": "observed (USGS 01311850, 01311875; parameter 62620)",
                "wind_speed_mps_daily_avg": w.get("wind_avg_mps"), "wind_gust_2min_mps_daily": w.get("wind_gust_2min_mps"),
                "precip_mm_daily": w.get("precip_mm"), "recent_precipitation_3d_prior_mm": w.get("precip_3d_prior_mm"),
                "weather_local_date": w.get("local_date"), "quality_flags": flags,
            })

        # Product 3: pooled series and one columnar record set per marsh polygon.
        pooled[name] = [{"acquisition_id": a, "date": d, "marsh_hh_db": float(mdb[t]), "upland_hh_db": float(udb[t]),
                         "marsh_relative_db": float(y[t]), "marsh_hv_db": num(m_hv[t]), "upland_hv_db": num(u_hv[t]),
                         "marsh_relative_hv_db": num(y_hv[t]), "mean_water_level_m_navd88": float(h[t]),
                         "passes_coverage_screen": bool(ok[t])}
                        for t, (a, d) in enumerate(zip(acq_ids, dates))]
        track_regions = []
        for k, feat in enumerate(tr["features"], start=1):
            idx = flat(tr["polygon_raster"] == k)
            if idx.size == 0:
                continue
            rid = f"osm_{feat['properties']['osm_id']}"
            hh, cnt = region_stats(db, idx)
            hv, _ = region_stats(hv_db, idx) if hv_db is not None else (nan, nan)
            cov = cnt / idx.size
            lk = (np.array([np.nanmedian(looks[t].ravel()[idx]) if np.isfinite(looks[t].ravel()[idx]).any() else np.nan
                            for t in range(n_t)]) if looks is not None else nan)
            use = np.isfinite(hh) & (cov >= COVERAGE_MIN) & (tr["cov_u"] >= COVERAGE_MIN)
            fit = {"model_status": "insufficient_evidence", "n": int(use.sum())}
            if use.sum() >= 10:
                lr = stats.linregress(h[use], (hh - udb)[use])
                fit = {"model_status": "exploratory", "n": int(use.sum()), "slope_db_per_m": float(lr.slope),
                       "pearson_r": float(lr.rvalue), "p_uncorrected": float(lr.pvalue)}
            track_regions.append({
                "region_id": rid, "kind": feat["properties"].get("wetland", ""), "n_pixels": int(idx.size),
                "hh_backscatter_db": [num(v) for v in hh], "hv_backscatter_db": [num(v) for v in hv],
                "marsh_relative_db": [num(v) for v in hh - udb], "valid_pixel_count": [int(c) for c in cnt],
                "coverage_fraction": [float(c) for c in cov], "number_of_looks_median": [num(v) for v in lk],
                "tide_response": fit,
            })
            gf = region_features.setdefault(rid, {"type": "Feature", "geometry": feat["geometry"], "properties": {
                "region_id": rid, "kind": feat["properties"].get("wetland", ""), "name": feat["properties"].get("name", ""),
                "source": "OpenStreetMap (not NWI)", "n_pixels_10m": int(idx.size), "tide_response": {}}})
            gf["properties"]["tide_response"][name] = fit
        regions[name] = {"acquisition_ids": acq_ids, "regions": track_regions}

        # Product 4: models and validation.
        variants = {"all_dates": np.arange(n_t)}
        if not ok.all():
            variants["coverage_screened"] = np.where(ok)[0]
        models[name] = {"excluded_by_screen": [d for d, k in zip(dates, ok) if not k],
                        "duplicate_granules_dropped": tr["dropped_duplicates"], "series": {}, "hv_secondary": {}}
        for vname, idx in variants.items():
            sub = [dates[i] for i in idx]
            a = analyse(sub, h[idx], y[idx])
            a["verdict"] = verdict(a)
            models[name]["series"][vname] = a
            good = idx[np.isfinite(y_hv[idx])]
            if len(good) >= 10:
                models[name]["hv_secondary"][vname] = {
                    "status": "secondary; not preregistered",
                    "marsh_minus_upland_hv": regress(y_hv[good], h[good]),
                    "marsh_hv_only": regress(m_hv[good], h[good])}
            b, adj, tn, ps = a["benchmark"], a["tide_adjustment"], a["trend"], a["persistence"]
            print(f"\n== {name} / {vname}  n={a['n']}  slope {a['regression']['slope_db_per_m']:+.2f} dB/m  -> {a['verdict']}")
            for split in ("chronological_half", "rolling_origin"):
                s = b[split]
                print(f"  {split:19s} RMSE const {s['constant']['rmse_db']:.2f}  seasonal {s['seasonal']['rmse_db']:.2f}  "
                      f"tide {s['tide']['rmse_db']:.2f}  (tide vs const {s['tide']['rmse_improvement_vs_constant']:+.0%}, "
                      f"n={s['n_test_predictions']})")
            ho = adj["held_out"]
            print(f"  held-out adjust     SD {ho['raw']['sd_db']:.2f} -> {ho['adjusted']['sd_db']:.2f} dB ({ho['sd_reduction']:+.0%}), "
                  f"r with h {ho['raw']['r_with_water_level']:+.2f} -> {ho['adjusted']['r_with_water_level']:+.2f}")
            print("  injected step refit " + ", ".join(
                f"+{r['injected_db']:.0f} -> {r['recovered_refit_beta_db']:+.2f}" for r in adj["injected_step"]["results"]))
            print(f"  trend (exploratory) gamma {tn['gamma_db_per_year']:+.2f} dB/yr "
                  f"CI [{tn['gamma_95ci_db_per_year'][0]:+.2f},{tn['gamma_95ci_db_per_year'][1]:+.2f}] p {tn['gamma_p']:.3f}")
            print(f"  persistence         longest run {ps['longest_same_sign_run']} (perm p {ps['longest_run_permutation_p']:.3f}), "
                  f"lag-1 r {ps['lag1_autocorrelation']:+.2f}, candidates "
                  + (", ".join(f"{c['start']}..{c['end']} {c['direction']} {c['median_residual_db']:+.2f}"
                               for c in ps["candidate_periods"]) or "none"))
            hvm = models[name]["hv_secondary"].get(vname)
            if hvm:
                m = hvm["marsh_minus_upland_hv"]
                print(f"  HV (secondary)      slope {m['slope_db_per_m']:+.2f} dB/m CI [{m['slope_ols_95ci'][0]:+.2f},"
                      f"{m['slope_ols_95ci'][1]:+.2f}] r {m['pearson_r']:+.2f} n={m['n']}")
        expl = [r["tide_response"] for r in track_regions if r["tide_response"]["model_status"] == "exploratory"]
        print(f"  regions             {len(track_regions)} with pixels, {len(expl)} with an exploratory fit, "
              f"{sum(e['slope_db_per_m'] < 0 for e in expl)} negative slopes")

    meta = {
        "y_definition": "median marsh HH (dB) minus median upland HH (dB); medians of per-pixel dB, "
                        "not dB of a linear-intensity aggregate",
        "h_definition": "mean of USGS 01311850 (Inwood) and 01311875 (Rockaway) at the overpass, m NAVD88, observed",
        "marsh_source": "OpenStreetMap wetland polygons (not NWI)",
        "upland_definition": "outside marsh polygons, temporal-median HH >= -5 dB (brightness proxy, not land cover)",
        "coverage_screen": f"marsh and upland valid fraction >= {COVERAGE_MIN}; chosen post hoc",
        "deduplication": "one record per physical acquisition; the highest processing version is kept",
        "orbits": "ascending and descending are modelled separately, each with its own intercept and slope",
        "region_minimum_valid_pixels": MIN_REGION_PIXELS,
        "pooled_marsh_level_only": True,
        "per_region_model_status": "exploratory; p-values uncorrected across regions",
        "verdict_rule": "strong = tide beats constant by >10% RMSE on both splits, held-out adjustment lowers SD and "
                        "|r with h|, and a refit keeps injected steps within 25%; mixed = 2-3 of 4; weak = fewer",
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "acquisitions.json": {"meta": {"deduplication": meta["deduplication"]}, "acquisitions": acquisitions},
        "environment.json": {"meta": {"weather_source": "NOAA daily summaries, JFK (USW00094789); full-day values",
                                      "h_definition": meta["h_definition"]}, "observations": environment},
        "timeseries.json": {"meta": meta, "pooled": pooled, "regions": regions},
        "models.json": {"meta": meta, "tracks": models},
        "regions.geojson": {"type": "FeatureCollection", "features": list(region_features.values())},
    }
    for fname, payload in payloads.items():
        (out / fname).write_text(json.dumps(payload, indent=1) + "\n")
    copied = []
    for src in (args.projects, bay / "hydrographs.json"):
        if src.exists():
            shutil.copyfile(src, out / src.name)
            copied.append(src.name)
    print(f"\nwrote {', '.join([*payloads, *copied])} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
