#!/usr/bin/env bash
# Load the analysis artifacts from data/interim into an R2 bucket.
#
#   ./seed.sh           -> LOCAL simulated R2 (no Cloudflare account needed)
#   ./seed.sh --remote  -> the real bucket (needs `wrangler login` first)
#
# Idempotent: re-running overwrites the same keys.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
source_dir="$repo_root/data/interim"
bucket="${R2_BUCKET:-groundtruth-dev}"

target="--local"
label="local simulated R2"
if [ "${1:-}" = "--remote" ]; then
  target="--remote"
  label="REMOTE bucket $bucket"
fi

cd "$here"

if [ ! -d "$source_dir" ]; then
  echo "No $source_dir. Run 'make fetch DATE=...' first." >&2
  exit 1
fi

if [ ! -f "$source_dir/index.json" ]; then
  echo "No index.json. Run 'make index' first." >&2
  exit 1
fi

echo "seeding $label from $source_dir"
echo ""

put() {
  local file="$1" key="$2" content_type="$3"
  local size
  size=$(wc -c < "$file" | tr -d ' ')
  printf '  %-12s %8s KiB  %s\n' "$content_type" "$((size / 1024))" "$key"
  npx wrangler r2 object put "$bucket/$key" \
    --file="$file" \
    --content-type="$content_type" \
    $target >/dev/null
}

put "$source_dir/index.json" "data/index.json" "application/json"

shopt -s nullglob
for tif in "$source_dir"/*.tif; do
  name="$(basename "$tif")"
  [ "$name" = "selftest.tif" ] && continue
  put "$tif" "data/$name" "image/tiff"
done

for provenance in "$source_dir"/*.provenance.json; do
  name="$(basename "$provenance")"
  put "$provenance" "data/$name" "application/json"
done

echo ""
echo "done. Start the Worker with:"
if [ "$target" = "--local" ]; then
  echo "    cd cloudflare && npx wrangler dev"
else
  echo "    cd cloudflare && npx wrangler deploy"
fi
