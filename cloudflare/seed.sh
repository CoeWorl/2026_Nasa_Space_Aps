#!/usr/bin/env bash
# Load the GROUNDTRUTH artifacts into an R2 bucket: the analysis rasters,
# the small JSON/GeoJSON data products, and the explore.html page.
#
#   ./seed.sh           -> LOCAL simulated R2 (no Cloudflare account needed)
#   ./seed.sh --remote  -> the real bucket (needs `wrangler login` first)
#
# Each part is seeded only if its source exists, so the explore app can be
# deployed without rasters, and rasters without the app. Idempotent:
# re-running overwrites the same keys.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
raster_dir="$repo_root/data/interim"
app_dir="$repo_root/data/processed/jamaica_bay/app"
page="$repo_root/web/explore.html"
bucket="${R2_BUCKET:-groundtruth-dev}"

target="--local"
label="local simulated R2"
if [ "${1:-}" = "--remote" ]; then
  target="--remote"
  label="REMOTE bucket $bucket"
fi

cd "$here"

echo "seeding $label"
echo ""

put() {
  local file="$1" key="$2" content_type="$3"
  local size
  size=$(wc -c < "$file" | tr -d ' ')
  printf '  %-18s %8s KiB  %s\n' "$content_type" "$((size / 1024))" "$key"
  npx wrangler r2 object put "$bucket/$key" \
    --file="$file" \
    --content-type="$content_type" \
    $target >/dev/null
}

seeded_any=0
shopt -s nullglob

# 1. Analysis rasters (optional). Needs `make fetch` then `make index`.
if [ -f "$raster_dir/index.json" ]; then
  echo "rasters from $raster_dir"
  put "$raster_dir/index.json" "data/index.json" "application/json"
  for tif in "$raster_dir"/*.tif; do
    name="$(basename "$tif")"
    [ "$name" = "selftest.tif" ] && continue
    put "$tif" "data/$name" "image/tiff"
  done
  for provenance in "$raster_dir"/*.provenance.json; do
    put "$provenance" "data/$(basename "$provenance")" "application/json"
  done
  seeded_any=1
  echo ""
else
  echo "skip rasters: no $raster_dir/index.json (run 'make fetch DATE=...' then 'make index')"
  echo ""
fi

# 2. Data products for explore.html (optional). Served at /data-app/<name>.
if [ -d "$app_dir" ]; then
  echo "data products from $app_dir"
  for f in "$app_dir"/*.json "$app_dir"/*.geojson; do
    name="$(basename "$f")"
    case "$name" in
      *.geojson) ct="application/geo+json" ;;
      *)         ct="application/json" ;;
    esac
    put "$f" "data/app/$name" "$ct"
    seeded_any=1
  done
  echo ""
else
  echo "skip products: no $app_dir (run tide_adjust.py --out-dir data/processed/jamaica_bay/app)"
  echo ""
fi

# 3. The explore.html page (optional). Served by the Worker at /explore.html.
if [ -f "$page" ]; then
  echo "page from $page"
  put "$page" "app/explore.html" "text/html"
  seeded_any=1
  echo ""
fi

if [ "$seeded_any" -eq 0 ]; then
  echo "nothing to seed: no rasters, products, or page found." >&2
  exit 1
fi

echo "done. Start the Worker with:"
if [ "$target" = "--local" ]; then
  echo "    cd cloudflare && npx wrangler dev        # then open /explore.html"
else
  echo "    cd cloudflare && npx wrangler deploy"
fi
