"""Figures for the Jamaica Bay tide-backscatter results.

    python scripts/make_figures.py --out-dir DIR

Reads only saved results (tide_regression.json, tide_qc.json). No new analysis.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path("data/processed/jamaica_bay")
DESC = BASE / "tide_regression.json"
ASC = BASE / "track133/tide_regression.json"
QC = BASE / "qc/tide_qc.json"


def load_per_date(path: Path):
    per = json.loads(path.read_text())["per_date"]
    return (
        [date.fromisoformat(p["date"]) for p in per],
        np.array([p["h_m"] for p in per]),
        np.array([p["marsh_db"] for p in per]),
        np.array([p["upland_db"] for p in per]),
        np.array([p["marsh_minus_upland_db"] for p in per]),
    )


def fig_scatter(out: Path) -> None:
    _, hd, _, _, yd = load_per_date(DESC)
    _, ha, _, _, ya = load_per_date(ASC)
    qc = json.loads(QC.read_text())
    screened_dates = {r["date"] for r in qc["tracks"]["133A022_ascending"]["coverage_by_date"] if r["screened"]}
    asc_dates = [p["date"] for p in json.loads(ASC.read_text())["per_date"]]
    mask_s = np.array([d in screened_dates for d in asc_dates])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(hd, yd, s=42, color="#1f6f8b", label="Descending (069), 26 dates")
    ax.scatter(ha[~mask_s], ya[~mask_s], s=60, facecolors="none", edgecolors="#c0392b",
               linewidths=1.6, label="Ascending (133), excluded partial-coverage dates")
    ax.scatter(ha[mask_s], ya[mask_s], s=42, color="#e67e22", label="Ascending (133), 23 screened dates")
    for name, x, y, color in [("desc", hd, yd, "#1f6f8b"), ("asc screened", ha[mask_s], ya[mask_s], "#e67e22")]:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, a + b * xs, color=color, lw=1.6, ls="--", label=f"fit: {name} slope {b:+.2f} dB/m")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("Water level at overpass (m, NAVD88, mean of two USGS gauges)")
    ax.set_ylabel("Marsh minus upland HH backscatter (dB)")
    ax.set_title("Jamaica Bay: marsh backscatter falls as water rises")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "01_tide_vs_backscatter.png", dpi=150)
    plt.close(fig)


def fig_timeseries(out: Path) -> None:
    dd, hd, md, ud, yd = load_per_date(DESC)
    da, ha, ma, ua, ya = load_per_date(ASC)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(dd, hd, "o-", color="#2c3e50", ms=4, label="Descending water level")
    ax1.plot(da, ha, "s-", color="#7f8c8d", ms=4, label="Ascending water level")
    ax1.set_ylabel("Water level (m NAVD88)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    ax2.plot(dd, yd, "o-", color="#1f6f8b", ms=4, label="Descending marsh minus upland (dB)")
    ax2.plot(da, ya, "s-", color="#e67e22", ms=4, label="Ascending marsh minus upland (dB)")
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_ylabel("Marsh minus upland (dB)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.suptitle("Time series: backscatter tracks water level, not a trend")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "02_timeseries.png", dpi=150)
    plt.close(fig)


def fig_coverage(out: Path) -> None:
    qc = json.loads(QC.read_text())
    rows = qc["tracks"]["133A022_ascending"]["coverage_by_date"]
    d = [date.fromisoformat(r["date"]) for r in rows]
    cm = np.array([r["marsh_cov"] for r in rows])
    cu = np.array([r["upland_cov"] for r in rows])
    scr = np.array([r["screened"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(d, cm, width=6, color=np.where(scr, "#2e86ab", "#c0392b"), label="marsh coverage")
    ax.plot(d, cu, "o", color="#27ae60", ms=5, label="upland coverage")
    ax.axhline(0.9, color="#333", ls="--", lw=1, label="90% screen (post hoc)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of pixels valid")
    ax.set_title("Ascending (track 133): coverage per acquisition. Red = excluded")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "03_ascending_coverage.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_scatter(args.out_dir)
    fig_timeseries(args.out_dir)
    fig_coverage(args.out_dir)
    for p in sorted(args.out_dir.glob("*.png")):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
