#!/usr/bin/env bash
# Black-box probe of the groundtruth-api Worker. Prints one line per check
# so behaviour can be compared across local `wrangler dev` and deployed Worker.
#
#   ./probe.sh                      -> http://localhost:8787
#   ./probe.sh https://example.dev  -> any base URL
#
# Expects the object named in T to exist (run ./seed.sh first).

set -uo pipefail
BASE="${1:-http://localhost:8787}"
T="${T:-NISAR_L2_PR_GCOV_032_069_D_068_4005_DHDH_A_20261004T235223_20261004T235258_P05023_N_F_J_001_frequencyA_HH.tif}"
U="$BASE/data/$T"

status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
header() {
  local name="$1"; shift
  curl -s -D - -o /dev/null "$@" | tr -d '\r' | grep -i "^$name:" | cut -d' ' -f2-
}

check() { printf '%-42s %s\n' "$1" "$2"; }

check "health" "$(status "$BASE/health")"
check "GET full object: status" "$(status "$U")"
check "GET full object: Content-Length" "$(header content-length "$U")"
check "HEAD: Accept-Ranges" "$(curl -sI "$U" | tr -d '\r' | grep -i '^accept-ranges:' | cut -d' ' -f2)"
check "HEAD: ETag" "$(curl -sI "$U" | tr -d '\r' | grep -i '^etag:' | cut -d' ' -f2-)"
check "HEAD: Cache-Control" "$(curl -sI "$U" | tr -d '\r' | grep -i '^cache-control:' | cut -d' ' -f2-)"
check "range bytes=0-99: status" "$(status -H 'Range: bytes=0-99' "$U")"
check "range bytes=0-99: Content-Range" "$(header content-range -H 'Range: bytes=0-99' "$U")"
check "range bytes=-64 (suffix): Content-Range" "$(header content-range -H 'Range: bytes=-64' "$U")"
check "range bytes=N- (open ended): status" "$(status -H 'Range: bytes=8312000-' "$U")"
check "range past EOF: status" "$(status -H 'Range: bytes=99999999-' "$U")"
check "range multi (0-1,5-6): status" "$(status -H 'Range: bytes=0-1,5-6' "$U")"
check "range garbage: status" "$(status -H 'Range: items=0-1' "$U")"
check "missing key: status" "$(status "$BASE/data/nope.tif")"
check "path traversal: status" "$(status "$BASE/data/..%2F..%2Fx.tif")"
check "disallowed suffix (.html): status" "$(status "$BASE/data/index.html")"
check "unknown prefix (/secret/x.tif): status" "$(status "$BASE/secret/x.tif")"
check "POST: status" "$(status -X POST "$U")"
check "OPTIONS preflight: status" "$(status -X OPTIONS "$U")"
check "CORS Allow-Origin" "$(header access-control-allow-origin "$U")"
