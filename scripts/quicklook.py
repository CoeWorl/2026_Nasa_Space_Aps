"""Render a GeoTIFF to a grayscale PNG you can open in any image viewer.

A development aid, not a deliverable. It exists so a radar raster can be
eyeballed in seconds without QGIS, a browser, or the frontend being ready.

It also prints the decibel distribution, which is the cheapest check on
whether the reader is interpreting the product correctly: over a city,
water should be very dark (roughly -20 dB and below) and built-up areas
bright (around 0 dB and above). If that structure is absent, something
upstream is wrong.

    make quicklook

PNG is written with the standard library, so this adds no dependency and
cannot be broken by a GDAL driver that only supports CreateCopy.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import rasterio

INTERIM = Path("/data/interim")
# Percentile stretch. SAR intensity has a long bright tail (corner
# reflectors, building faces); a min/max stretch would leave everything
# else black.
LOW_PERCENTILE, HIGH_PERCENTILE = 2.0, 98.0


def write_png_gray_alpha(path: Path, gray: np.ndarray, alpha: np.ndarray) -> None:
    """8-bit grayscale + alpha PNG. Invalid pixels become transparent."""
    height, width = gray.shape
    interleaved = np.empty((height, width, 2), dtype=np.uint8)
    interleaved[..., 0] = gray
    interleaved[..., 1] = alpha

    raw = b"".join(
        b"\x00" + interleaved[row].tobytes() for row in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    # Colour type 4 = grayscale with alpha.
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 4, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def render(tif_path: Path) -> Path | None:
    with rasterio.open(tif_path) as src:
        values = src.read(1).astype(np.float64)
        quantity = src.tags().get("quantity", "unknown")
        crs = src.crs

    finite = np.isfinite(values)
    if not finite.any():
        print(f"{tif_path.name}: no valid pixels, skipped")
        return None

    if quantity == "linear_intensity":
        usable = finite & (values > 0)
        display = np.full(values.shape, np.nan)
        display[usable] = 10.0 * np.log10(values[usable])
        unit = "dB"
    else:
        # Already a difference or an unknown quantity: show it as stored.
        usable = finite
        display = np.where(usable, values, np.nan)
        unit = quantity

    sample = display[usable]
    if sample.size == 0:
        print(f"{tif_path.name}: nothing to display, skipped")
        return None

    low, high = np.percentile(sample, [LOW_PERCENTILE, HIGH_PERCENTILE])
    print(f"\n{tif_path.name}")
    print(f"  quantity   {quantity} -> displayed in {unit}")
    print(f"  crs        {crs}")
    print(f"  size       {display.shape[1]} x {display.shape[0]} px")
    print(f"  valid      {usable.sum() / usable.size:.1%}")
    print(
        f"  {unit:<10} min {sample.min():7.2f}   p2 {low:7.2f}   "
        f"median {np.median(sample):7.2f}   p98 {high:7.2f}   "
        f"max {sample.max():7.2f}"
    )

    if high <= low:
        print("  degenerate stretch, skipped")
        return None

    scaled = np.clip((display - low) / (high - low), 0.0, 1.0)
    gray = np.nan_to_num(scaled * 255.0).astype(np.uint8)
    alpha = np.where(usable, 255, 0).astype(np.uint8)

    out_path = tif_path.with_suffix(".png")
    write_png_gray_alpha(out_path, gray, alpha)
    print(f"  wrote      {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or sorted(INTERIM.glob("*.tif"))
    if not targets:
        print(f"No GeoTIFFs found in {INTERIM}. Run 'make fetch DATE=...' first.")
        return 1

    written = [p for p in (render(t) for t in targets) if p is not None]
    print(f"\n{len(written)} PNG(s) written next to their GeoTIFFs.")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
